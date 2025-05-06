import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
import numpy as np
import scipy.sparse as sp

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.spmm(input, self.weight) if input.is_sparse else torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias if self.bias is not None else output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"

class MINE(nn.Module):
    def __init__(self, input_size_x, input_size_y, hidden_size):
        super(MINE, self).__init__()
        self.fc1_x = nn.Linear(input_size_x, hidden_size)
        self.fc1_y = nn.Linear(input_size_y, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        h = self.relu(self.fc1_x(x) + self.fc1_y(y))
        output = self.fc2(h)
        return output

class MIGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, device=None, use_mi_regularization=1):

        super(MIGCN, self).__init__()

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay if with_relu else 0
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.use_mi_regularization = use_mi_regularization

        if self.use_mi_regularization:
            self.mine_network = MINE(input_size_x=1, input_size_y=nclass, hidden_size=16).to(device)
            self.mine_optimizer = optim.Adam(self.mine_network.parameters(), lr=self.lr)

        self.identity_adj = None
        self.att_walk_adj = None
        self.original_adj = None

    def create_identity_adj(self, size):
        return torch.eye(size, device=self.device).to_sparse()

    def forward(self, x, att_walk_adj):
        if att_walk_adj is None:
            att_walk_adj = self.adj_norm

        if self.with_relu:
            x = F.relu(self.gc1(x, self.att_walk_adj))
        else:
            x = self.gc1(x, self.att_walk_adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, att_walk_adj)

        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        if self.use_mi_regularization:
            self.mine_network.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200,
            initialize=True, verbose=False, normalize=True, patience=500, **kwargs):

        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        if utils.is_sparse_tensor(features):
            features = features.to_dense()
        self.features = features.to(self.device)
        self.adj = adj.to(self.device)
        self.original_adj = adj.to(self.device)
        self.labels = labels.to(self.device)

        if initialize:
            self.initialize()

        if normalize:
            self.adj_norm = utils.normalize_adj_tensor(self.adj, sparse=utils.is_sparse_tensor(self.adj))
        else:
            self.adj_norm = self.adj

        num_nodes = adj.size(0)
        self.identity_adj = self.create_identity_adj(num_nodes)
        self.att_walk_adj = self.adj_norm.clone().detach()
        self.idx_train = idx_train
        self.idx_val = idx_val

        if self.use_mi_regularization:
            self._train_with_mi(train_iters, patience, verbose)
        else:
            self._train_without_mi(train_iters, patience, verbose)

    def compute_integrated_gradients(self, baseline, inputs, m=50):

        total_gradients = torch.zeros_like(inputs).to(self.device)
        steps = torch.linspace(0, 1, steps=m, device=self.device)

        delta = inputs - baseline

        for i in range(m):
            alpha = steps[i]
            scaled_input = baseline + alpha * delta
            scaled_input.requires_grad_(True)

            output = self.forward(scaled_input, self.att_walk_adj)
            loss = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])

            gradients = torch.autograd.grad(outputs=loss, inputs=scaled_input, retain_graph=True)[0]
            total_gradients += gradients

        integrated_gradients = delta * (total_gradients / m)
        return integrated_gradients

    def compute_mine_loss(self, x, y):

        joint = self.mine_network(x, y)
        y_shuffle = y[torch.randperm(y.size(0))]
        marginal = self.mine_network(x, y_shuffle)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def att_walk(self, max_idx, fea, edge_index, i, is_lil=False):

        adj_matrix = edge_index
        if not is_lil:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy(), edge_index[1].cpu().data.numpy()

        sim_matrix = self.estimated_similarity(max_idx, fea, adj_matrix, is_lil=is_lil)

        sim_matrix_norm = F.normalize(sim_matrix, p=1, dim=1)


        topk = 8
        topk_sim_matrix = torch.zeros_like(sim_matrix_norm)
        for j in range(sim_matrix_norm.size(1)):
            if sim_matrix_norm.size(0) < topk:
                actual_topk = sim_matrix_norm.size(0)
            else:
                actual_topk = topk
            topk_values, topk_indices = torch.topk(sim_matrix_norm[:, j], actual_topk)
            topk_sim_matrix[topk_indices, j] = sim_matrix_norm[topk_indices, j]

        new_adj_matrix = adj_matrix.to_dense().clone().cpu().numpy()


        for j in range(n_node):

            adj_edges = set(np.where(new_adj_matrix[:, j] > 0)[0])

            topk_edges = set(topk_sim_matrix[:, j].nonzero(as_tuple=True)[0].cpu().numpy())

            removed_edges = adj_edges - topk_edges
            num_removed = len(removed_edges)

            for edge in removed_edges:
                new_adj_matrix[edge, j] = 0

            potential_edges = topk_edges - adj_edges
            if len(potential_edges) > 0:
                sorted_potential = sorted(potential_edges, key=lambda x: sim_matrix_norm[x, j], reverse=True)
                edges_to_add = sorted_potential[:num_removed]
                for edge in edges_to_add:
                    new_adj_matrix[edge, j] = 1


        new_adj_matrix = torch.tensor(new_adj_matrix, dtype=torch.float32, device=self.device)

        new_adj_matrix.fill_diagonal_(1)

        row_sum = new_adj_matrix.sum(dim=1)
        row_sum_inv_sqrt = torch.pow(row_sum, -0.5)

        row_sum_inv_sqrt[torch.isinf(row_sum_inv_sqrt)] = 0

        D_inv_sqrt = row_sum_inv_sqrt.unsqueeze(1)
        new_adj_matrix_normalized = new_adj_matrix * D_inv_sqrt
        new_adj_matrix_normalized = new_adj_matrix_normalized * D_inv_sqrt.t()


        new_adj = new_adj_matrix_normalized.to_sparse().coalesce()

        if torch.isnan(new_adj.values()).any():
            print("Warning: NaN values detected in the adjacency matrix.")

        return new_adj

    def normalize_matrix(self, matrix):

        if matrix.is_sparse:
            matrix = matrix.to_dense()
        row_sum = matrix.sum(dim=0, keepdim=True)
        row_sum[row_sum == 0] = 1
        normalized_matrix = matrix / row_sum
        return normalized_matrix

    def estimated_similarity(self, max_idx, fea, edge_index, is_lil=False):

        adj_matrix = edge_index
        edge_index = adj_matrix.coalesce().indices() if not is_lil else torch.from_numpy(adj_matrix.tocoo().row).to(self.device)

        num_nodes = fea.shape[0]

        degree_matrix = self.calculate_degree_matrix(edge_index, num_nodes)
        degree_matrix_inv = self.inverse_sparse_matrix(degree_matrix)
        adj_matrix_dense = adj_matrix.to_dense()
        topo_pro = torch.matmul(degree_matrix_inv, adj_matrix_dense)
        att_pro = self.compute_att_pro(fea)

        alpha = 0.15
        beta = 1

        transition_matrix = (1 - beta) * topo_pro + beta * torch.tensor(att_pro, dtype=torch.float32, device=self.device)


        sensitive_attrs = fea[:, max_idx].unsqueeze(1)
        same_attr = sensitive_attrs == sensitive_attrs.t()
        A = same_attr.sum(dim=1).clamp(min=1).float()
        B = (~same_attr).sum(dim=1).clamp(min=1).float()

        A = A + 1e-10
        B = B + 1e-10

        adjustment = B / A
        adjustment_matrix = same_attr.float() * adjustment.unsqueeze(1) + (~same_attr).float() * (A / B).unsqueeze(1)
        transition_matrix = transition_matrix * adjustment_matrix

        transition_matrix = F.normalize(transition_matrix, p=1, dim=1)

        N = 10
        S_estimated = self.attributed_random_walk(N, num_nodes, alpha, transition_matrix)

        return S_estimated

    def attributed_random_walk(self, N, num_nodes, alpha, transition_matrix):

        S_estimated = alpha * torch.eye(num_nodes, device=self.device)
        att_random_walk = S_estimated.clone()

        for _ in range(N):
            att_random_walk = (1 - alpha) * torch.matmul(transition_matrix, att_random_walk)
            S_estimated += att_random_walk

        return S_estimated

    def inverse_sparse_matrix(self, sparse_matrix):

        dense_matrix = sparse_matrix.to_dense()
        dense_inverse = torch.inverse(dense_matrix)
        return dense_inverse

    def calculate_degree_matrix(self, edge_index, num_nodes):

        degrees = torch.bincount(edge_index[0], minlength=num_nodes).float().to(self.device) + 1  # 加1是为了自环

        indices = torch.stack([torch.arange(num_nodes, device=self.device), torch.arange(num_nodes, device=self.device)])
        sparse_degree_matrix = torch.sparse_coo_tensor(indices, degrees, size=(num_nodes, num_nodes), dtype=torch.float32)
        return sparse_degree_matrix

    def compute_att_pro(self, fea):

        inner_products = torch.mm(fea, fea.t())
        row_sum = inner_products.sum(dim=1, keepdim=True)
        normalized_weights = inner_products / (row_sum + 1e-10)
        return normalized_weights.detach().cpu().numpy().astype(np.float32)

    def convert_to_sparse_tensor(self, adj_matrix):

        adj_sparse = sp.coo_matrix(adj_matrix)
        indices = torch.from_numpy(
            np.vstack((adj_sparse.row, adj_sparse.col)).astype(np.int64)
        ).to(self.device)
        values = torch.from_numpy(adj_sparse.data.astype(np.float32)).to(self.device)
        shape = torch.Size(adj_matrix.shape)
        adj_tensor = torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float32).coalesce().to(self.device)
        return adj_tensor

    def reset_mine_parameters(self):

        def reset_fn(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.mine_network.apply(reset_fn)
        if hasattr(self, 'mine_optimizer'):
            self.mine_optimizer = optim.Adam(self.mine_network.parameters(), lr=self.lr)

    def _train_with_mi(self, train_iters, patience, verbose):

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lambda_mi = 0.2
        m = 50
        best_loss_val = float('inf')
        patience_counter = 0
        best_weights = deepcopy(self.state_dict())

        for epoch in range(train_iters):
            optimizer.zero_grad()
            print(f"Epoch {epoch}")

            baseline = torch.zeros_like(self.features).to(self.device)

            integrated_gradients = self.compute_integrated_gradients(baseline, self.features, m=m)

            attribution_norms = torch.norm(integrated_gradients, p=2, dim=0)
            max_idx = torch.argmax(attribution_norms).item()

            self.att_walk_adj = self.att_walk(max_idx, self.features, self.original_adj, i=1, is_lil=False)

            output = self.forward(self.features, self.att_walk_adj)
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])

            probs = F.softmax(output, dim=1)

            mi_estimate = self.compute_mine_loss(
                integrated_gradients[:, max_idx].unsqueeze(1)[self.idx_train],
                probs[self.idx_train]
            )

            total_loss = loss_train + lambda_mi * mi_estimate

            for param in self.mine_network.parameters():
                param.requires_grad = False

            total_loss.backward()

            for param in self.mine_network.parameters():
                param.requires_grad = True

            optimizer.step()

            for _ in range(80):
                self.mine_optimizer.zero_grad()
                mi_estimate = self.compute_mine_loss(
                    integrated_gradients[:, max_idx].unsqueeze(1)[self.idx_train].detach(),
                    probs[self.idx_train].detach()
                )
                mi_loss = -mi_estimate
                mi_loss.backward()
                self.mine_optimizer.step()

            if self.idx_val is not None:
                self.eval()
                with torch.no_grad():
                    val_output = self.forward(self.features, self.att_walk_adj)
                    loss_val = F.nll_loss(val_output[self.idx_val], self.labels[self.idx_val])

                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    patience_counter = 0
                    best_weights = deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f'早停机制在第 {epoch} 轮触发')
                    break
                self.train()

        if self.idx_val is not None:
            self.load_state_dict(best_weights)

    def _train_without_mi(self, train_iters, patience, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = float('inf')
        patience_counter = 0
        best_weights = deepcopy(self.state_dict())

        for epoch in range(train_iters):
            optimizer.zero_grad()
            if verbose:
                print(f"Epoch {epoch}")

            output = self.forward(self.features, self.att_walk_adj)
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])

            loss_train.backward()
            optimizer.step()


            if self.idx_val is not None:
                self.eval()
                with torch.no_grad():
                    val_output = self.forward(self.features, self.att_walk_adj)
                    loss_val = F.nll_loss(val_output[self.idx_val], self.labels[self.idx_val])

                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    patience_counter = 0
                    best_weights = deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f'早停机制在第 {epoch} 轮触发')
                    break
                self.train()

        if self.idx_val is not None:
            self.load_state_dict(best_weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("result:",
              f"loss= {loss_test.item():.4f}",
              f"accuracy= {acc_test.item():.4f}")
        return acc_test, output

    def predict(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.att_walk_adj)
        else:
            if not isinstance(adj, torch.Tensor):
                features, adj = utils.to_tensor(features, adj, device=self.device)

            if utils.is_sparse_tensor(features):
                features = features.to_dense()
            features = features.to(self.device)
            adj = adj.to(self.device)

            att_walk_adj = utils.normalize_adj_tensor(adj, sparse=utils.is_sparse_tensor(adj))
            return self.forward(features, att_walk_adj)
