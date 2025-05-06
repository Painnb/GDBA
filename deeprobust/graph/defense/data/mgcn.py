import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy

class GraphConvolution(Module):
    """简单的 GCN 层，类似于 https://github.com/tkipf/pygcn"""

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
        """图卷积层的前向传播函数"""
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)  # 处理稀疏输入
        else:
            support = torch.mm(input, self.weight)    # 处理稠密输入
        output = torch.spmm(adj, support)             # 邻接矩阵通常是稀疏的
        if self.bias is not None:
            return output + self.bias
        else:
            return output

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
    """包含可选互信息正则化的两层图卷积网络。"""

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, device=None, use_mi_regularization=1):

        super(MIGCN, self).__init__()

        assert device is not None, "请指定 'device' 参数！"
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

    def forward(self, x, adj):
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """初始化 GCN 层的参数。"""
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
        """训练 MIGCN 模型。"""

        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        if utils.is_sparse_tensor(features):
            features = features.to_dense()
        self.features = features.to(self.device)
        self.adj = adj.to(self.device)
        self.labels = labels.to(self.device)

        if initialize:
            self.initialize()

        # 规范化邻接矩阵
        if normalize:
            self.adj_norm = utils.normalize_adj_tensor(self.adj, sparse=utils.is_sparse_tensor(self.adj))
        else:
            self.adj_norm = self.adj

        self.idx_train = idx_train
        self.idx_val = idx_val

        if self.use_mi_regularization:
            self._train_with_mi(train_iters, patience, verbose)
        else:
            self._train_without_mi(train_iters, patience, verbose)

    def compute_integrated_gradients(self, baseline, inputs, m=50):
        total_gradients = torch.zeros_like(inputs)
        for k in range(1, m + 1):
            alpha = float(k) / m
            interpolated_input = baseline + alpha * (inputs - baseline)
            interpolated_input.requires_grad_(True)
            output = self.forward(interpolated_input, self.adj_norm)
            loss = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            gradients = torch.autograd.grad(outputs=loss, inputs=interpolated_input, create_graph=False)[0]
            total_gradients += gradients
        return (inputs - baseline) * (total_gradients / m)

    def compute_mine_loss(self, x, y):
        joint = self.mine_network(x, y)
        y_shuffle = y[torch.randperm(y.size(0))]
        marginal = self.mine_network(x, y_shuffle)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def _train_with_mi(self, train_iters, patience, verbose):
        self.train()
        # 使用全部参数，包括主网络和 MINE 网络
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lambda_mi = 0.2  # 控制互信息项的权重
        m = 50  # 插值步数
        best_loss_val = float('inf')
        patience_counter = 0

        for i in range(train_iters):
            optimizer.zero_grad()

            # 主网络的前向传播
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])

            # 定义基线输入（全零向量）
            baseline = torch.zeros_like(self.features)

            # 计算集成梯度
            integrated_gradients = self.compute_integrated_gradients(baseline, self.features, m=m)

            # 计算归因值的范数，并选择最重要的特征
            attribution_norms = torch.norm(integrated_gradients, p=2, dim=0)
            max_idx = torch.argmax(attribution_norms)
            max_attr_feature = self.features[:, max_idx].unsqueeze(1)

            # 获取预测概率
            probs = F.softmax(output, dim=1)

            # 训练 MINE 网络
            for _ in range(10):  # MINE 网络训练步数
                self.mine_optimizer.zero_grad()
                mi_estimate = self.compute_mine_loss(
                    max_attr_feature[self.idx_train].detach(),
                    probs[self.idx_train].detach()
                )
                mi_loss = -mi_estimate  # 最大化互信息
                mi_loss.backward()
                self.mine_optimizer.step()

            # 使用 MINE 网络计算互信息估计值，并将其加入主网络的损失
            mi_estimate = self.compute_mine_loss(
                max_attr_feature[self.idx_train],
                probs[self.idx_train]
            )

            # 将互信息项加入到总损失中
            total_loss = loss_train + lambda_mi * mi_estimate

            # 冻结 MINE 网络的参数，防止梯度回传到 MINE 网络
            for param in self.mine_network.parameters():
                param.requires_grad = False

            # 对主网络的参数进行反向传播
            total_loss.backward()

            # 恢复 MINE 网络的参数的 requires_grad 属性
            for param in self.mine_network.parameters():
                param.requires_grad = True

            optimizer.step()

            if verbose and i % 10 == 0:
                print(f'Epoch {i}, training loss: {loss_train.item():.4f}, MI estimate: {mi_estimate.item():.4f}')

            # 验证集评估和早停机制
            if self.idx_val is not None:
                self.eval()
                with torch.no_grad():
                    output = self.forward(self.features, self.adj_norm)
                    loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])

                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    patience_counter = 0
                    best_weights = deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {i}')
                    break
                self.train()

        # 加载最佳模型参数
        if self.idx_val is not None:
            self.load_state_dict(best_weights)

    def _train_without_mi(self, train_iters, patience, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = float('inf')
        patience_counter = 0

        for epoch in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, training loss: {loss_train.item():.4f}')

            # 验证集评估和早停机制
            if self.idx_val is not None:
                self.eval()
                with torch.no_grad():
                    output = self.forward(self.features, self.adj_norm)
                    loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])

                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    patience_counter = 0
                    best_weights = deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}')
                    break
                self.train()

        # 加载最佳模型参数
        if self.idx_val is not None:
            self.load_state_dict(best_weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("测试集结果:",
              f"loss= {loss_test.item():.4f}",
              f"accuracy= {acc_test.item():.4f}")
        return acc_test, output

    def predict(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if not isinstance(adj, torch.Tensor):
                features, adj = utils.to_tensor(features, adj, device=self.device)

            if utils.is_sparse_tensor(features):
                features = features.to_dense()
            features = features.to(self.device)
            adj = adj.to(self.device)

            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(features, adj_norm)
