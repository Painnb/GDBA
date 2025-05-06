import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import to_dense_adj
from deeprobust.graph import utils
import torch.optim as optim

class GINLayer(Module):
    """Graph Isomorphism Network Layer"""

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GINLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(out_features, out_features)
        )
        self.conv = GINConv(self.mlp)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        return self.conv(x, edge_index)

class RGIN(Module):
    """Robust Graph Isomorphism Network Against Adversarial Attacks.

    Parameters
    ----------
    nnodes : int
        number of nodes in the input graph
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    lr : float
        learning rate for GIN
    dropout : float
        dropout rate for GIN
    device: str
        'cpu' or 'cuda'.
    """

    def __init__(self, nnodes, nfeat, nhid, nclass, lr=0.01, dropout=0.6, device='cpu'):
        super(RGIN, self).__init__()

        self.device = device
        self.lr = lr
        self.nclass = nclass
        self.gc1 = GINLayer(nfeat, nhid, dropout=dropout)
        self.gc2 = GINLayer(nhid, nclass, dropout=dropout)
        self.dropout = dropout
        self.adj, self.features, self.labels = None, None, None

    def forward(self):
        x, edge_index = self.features, self.edge_index
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, verbose=True, **kwargs):
        """Train RGIN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GIN training process will not adopt early stopping
        train_iters : int
            number of training epochs
        verbose : bool
            whether to show verbose logs
        """

        adj, features, labels = utils.to_tensor(adj.todense(), features.todense(), labels, device=self.device)
        adj_sparse = adj.to_sparse()  # 转换为稀疏张量
        edge_index = adj_sparse.coalesce().indices()  # 获取 edge_index

        self.features, self.edge_index, self.labels = features, edge_index, labels

        print('=== training rgin model ===')
        self._initialize()
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward()
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward()
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output

        print('=== picking the best model according to the performance on validation ===')

    def test(self, idx_test):
        """Evaluate the performance on test set"""
        self.eval()
        output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        return acc_test, output

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of RGIN
        """
        self.eval()
        return self.forward()

    def _loss(self, input, labels):
        loss = F.nll_loss(input, labels)
        return loss

    def _initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

if __name__ == "__main__":

    from deeprobust.graph.data import PrePtbDataset, Dataset
    # load clean graph data
    dataset_str = 'pubmed'
    data = Dataset(root='/tmp/', name=dataset_str, seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # load perturbed graph data
    perturbed_data = PrePtbDataset(root='/tmp/', name=dataset_str)
    perturbed_adj = perturbed_data.adj

    # train defense model
    model = RGIN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1],
                         nclass=labels.max()+1, nhid=32, device='cuda').to('cuda')
    model.fit(features, perturbed_adj, labels, idx_train, idx_val,
                      train_iters=200, verbose=True)
    model.test(idx_test)

    prediction_1 = model.predict()
    print(prediction_1)