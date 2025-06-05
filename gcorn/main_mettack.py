"""
Contains the complete implementation to reproduce the results of the "Mettack"
---
The implementation contains all the related benchmarks:
    - GCNGuard
    - RGCN
    - GCN-Jaccard
    - Our proposed Noisy GCN.

To use the benchmarks (GCNGuard, RGCN ...), please adapt the argument "defense"
in the "test" function. We provided an example of their use in the main section
of this file.
"""
import os
import torch
import numpy as np
import scipy
import argparse
import torch.nn.functional as F
import torch.optim as optim
import copy
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import MetaApprox, Metattack
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_scipy_sparse_matrix,to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from deeprobust.graph.defense import *
from deeprobust.graph.defense.ours_gcn import Our_GCN
from deeprobust.graph.defense.ours_gin import Our_GIN
from deeprobust.graph.defense.gcn import GCN
from deeprobust.graph.defense.gin import GIN
# from deeprobust.graph.defense_pyg import GCN
from ogb.nodeproppred import PygNodePropPredDataset
from deeprobust.graph.data import Pyg2Dpr
# from deeprobust.graph.defense.noisy_gcn import Noisy_GCN
from torch.utils.data import DataLoader
from gcorn import GCORN
from gcorn import parseval_weight_projections, test_gcorn, train_gcorn, compute_acc_perturbation
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='uai', choices=['cora', 'citeseer', 'acm', 'blogcatalog', 'uai', 'flickr'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,help='whether use symmetric matrix')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')
parser.add_argument('--modelname', type=str, default='GCN', choices=['Our_GCN','Our_GIN','GCN','GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard', choices=['GCNJaccard', 'RGCN', 'GCNSVD']) 
parser.add_argument('--GNNGuard', type=bool, default=False, choices=[True, False])

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: %s' % device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# Load the Dataset
if args.dataset in ['cora', 'citeseer', 'acm', 'blogcatalog', 'uai', 'flickr']:
    data = Dataset(root='Datasets/', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

elif args.dataset in ['ogbn-arxiv']:
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv')
    data = Pyg2Dpr(dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

elif args.dataset in ['film']:
    dataset_split = '../hetero_data/pei_data_split/splits/' + str(args.dataset) + '_split_0.6_0.2_' + str(
            0) + '.npz'
    data = load_data(args.dataset, dataset_split)
    features = data.x.cpu().detach().numpy()
    print(features)
    features = sp.csr_matrix(features)
    adj, labels, idx_train, idx_val, idx_test = data.adj_origin, data.y, data.train_mask, data.val_mask, data.test_mask

elif args.dataset in ['texas', 'cornell']:
    dataset = DataLoader(args.dataset)
    data = dataset[0]
    features = data.x
    labels = data.y
    adj = to_scipy_sparse_matrix(data.edge_index)
    # 节点数量
    num_nodes = data.num_nodes

    # 生成训练、验证、测试索引
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    idx_train = indices[:train_size]
    idx_val = indices[train_size:train_size + val_size]
    idx_test = indices[train_size + val_size:]

    # 转换为张量
    idx_train = torch.tensor(idx_train, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)
    idx_test = torch.tensor(idx_test, dtype=torch.long)

# 生成mask
num_nodes = features.shape[0]
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

idx_unlabeled = np.union1d(idx_val, idx_test)
if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)

# Transforming the perturbation rate into edges
perturbations = int(args.ptb_rate * (adj.sum()//2))

# Preprocessing and sparsifying the adjacency and the feature matrix
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
adj, features = csr_matrix(adj), csr_matrix(features)

# Transform to undirected adjacency (spacially useful for OGB Data)
adj = adj + adj.T
adj[adj>1] = 1


# Setup GCN as the Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
            dropout=0.5, with_relu=False, with_bias=False, weight_decay=5e-4,
                                                            device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, train_iters=201)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

# Initialize the Attack
if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0],
                        feature_shape=features.shape, attack_structure=True,
                        attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0],
                        feature_shape=features.shape,  attack_structure=True,
                        attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)




def test_noisy(adj):
    """
    Main function to test our proposed NoisyGCN
    ---
    Inputs:
        new_adj: the clean/perturbed adjacency to be tested

    Output:
        acc_test: The resulting accuracy test
    """


    best_acc_val = 0
    # We test the best noise value based on the validation nodes as specified
    # in the main paper
    for beta in np.arange(0, args.beta_max, args.beta_min):
        classifier = Noisy_GCN(nfeat=features.shape[1], nhid=16,
                                nclass=labels.max().item() + 1, dropout=0.5,
                                    device=device, noise_ratio_1=beta)

        classifier = classifier.to(device)

        classifier.fit(features, adj, labels, idx_train, train_iters=200,
                       idx_val=idx_val,
                       idx_test=idx_test,
                       verbose=False, attention=False)
        classifier.eval()

        # Validation Acc
        acc_val, _ = classifier.test(idx_val)

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            acc_test, _ = classifier.test(idx_test)

    return acc_test.item()


def test(adj, defense="GCN"):
    """
    Main function to test the considered benchmarks
    ---
    Inputs:
        adj: the clean/perturbed adjacency to be tested
        defense (str,): The considered defense method (Guard, Jaccard ..)

    Output:
        acc_test: The resulting accuracy test
    """

    if defense == "GCN":
        classifier = globals()[args.modelname](nfeat=features.shape[1],
            nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    elif defense == "Guard":
        classifier = globals()[args.modelname](nfeat=features.shape[1],
            nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = True

    elif defense == "Ours":
        classifier = globals()[args.modelname](nfeat=features.shape[1],
            nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = True

    elif defense == "GCNJaccard":
        if args.modelname == "GCN":
            classifier = GCNJaccard(nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        elif args.modelname == "GIN":
            classifier = GINJaccard(nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    elif defense == "GCNSVD":
        if args.modelname == "GCN":
            classifier = GCNSVD(nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        elif args.modelname == "GIN":
            classifier = GINSVD(nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    elif defense == "RGCN":
        if args.modelname == "GCN":
            classifier = RGCN(nnodes=adj.shape[0],nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        elif args.modelname == "GIN":
            classifier = RGIN(nnodes=adj.shape[0],nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    elif defense == "ProGNN":
        classifier = ProGNN(model, args, device)
        classifier.fit(features, adj, labels, idx_train, idx_val)
        acc_test = classifier.test(features, labels, idx_test)
        return acc_test

    elif defense == "GCORN":
        l_acc_GCORN = []
        model_ro = GCORN(features.shape[1], 16, labels.max().item() + 1).to(device)
        optimizer = torch.optim.Adam(model_ro.parameters(), lr = args.lr)
        best_val_acc = final_test_acc = 0
        for epoch in range(1, args.epochs + 1):
            model_ro, loss = train_gcorn(model_ro, optimizer, data, adj)
            train_acc, val_acc, tmp_test_acc = test_gcorn(model_ro, data, adj)
            # Chose the best model based on validation
            if val_acc > best_val_acc:
                best_model_ro = copy.deepcopy(model_ro)
                best_val_acc = val_acc
                test_acc = tmp_test_acc

        # Run accuracy on both normal data and attacked data
        acc_1, acc_2, h_1, h_2 = compute_acc_perturbation(best_model_ro,
                                                    data, adj, features)

        l_acc_GCORN.append(acc_1)
        l_acc_GCORN_attacked.append(acc_2)
        return l_acc_GCORN,l_acc_GCORN_attacked

    else:
        classifier = globals()[defense](nnodes=adj.shape[0], nhid=16,
                        nfeat=features.shape[1], nclass=labels.max().item() + 1,
                                                    dropout=0.5, device=device)
        attention = False

    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=201,idx_val=idx_val, idx_test=idx_test, verbose=False, attention=attention)
    classifier.eval()

    acc_test, _ = classifier.test(idx_test)
    return acc_test.item()


def test_GCORN(adj, attack_adj, defense="GCN"):
    l_acc_GCORN = []
    l_acc_GCORN_attacked = []
    model_ro = GCORN(features.shape[1], 16, labels.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model_ro.parameters(), lr = args.lr)
    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs + 1):
        model_ro, loss = train_gcorn(model_ro, optimizer, data, adj)
        train_acc, val_acc, tmp_test_acc = test_gcorn(model_ro, data, adj)
        # Chose the best model based on validation
        if val_acc > best_val_acc:
            best_model_ro = copy.deepcopy(model_ro)
            best_val_acc = val_acc
            test_acc = tmp_test_acc

    # Run accuracy on both normal data and attacked data
    acc_1, acc_2, h_1, h_2 = compute_acc_perturbation(best_model_ro,
                                                data, adj, attack_adj)

    l_acc_GCORN.append(acc_1)
    l_acc_GCORN_attacked.append(acc_2)
    return l_acc_GCORN, l_acc_GCORN_attacked


if __name__ == '__main__':
    """
    Main function containing the Mettack implementation, please note that you
    need to uncomment the last part to use the other benchamarks
    """
    output_file = "mettack_gcorn.csv"
    # 初始化文件并添加表头
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("dataset,ptb_rate,acc_gcn_attacked\n")
    # Apply the Attack and get the resulting adjacency
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations,
                                                            ll_constraint=False)
    modified_adj = model.modified_adj
    modified_adj_sparse = csr_matrix(modified_adj.cpu().numpy())


    # print('=== testing NoisyGCN ===')
    # attention=False
    # acc_noise_clean=test_noisy(adj)
    # acc_noise_attacked=test_noisy(modified_adj_sparse)

    # To run another defense:

    # --- Normal GCN --- #
    # print('=== testing Normal GCN ===')
    # acc_gcn_non_attacked = test(adj)
    # acc_gcn_attacked = test(modified_adj_sparse)
    # print('---------------')
    # print("GCN Non Attacked Acc - {}" .format(acc_gcn_non_attacked))
    # print("GCN Attacked Acc - {}" .format(acc_gcn_attacked))
    # print('---------------')


    # --- RGCN --- #
    # 测试baseline时，modelname选择GCN、GIN
    # print('=== testing RGCN ===')
    # attention = False
    # acc_rgcn_non_attacked = test(adj, defense = "RGCN")
    # acc_rgcn_attacked = test(modified_adj_sparse, defense = "RGCN")
    # print('---------------')
    # print("RGCN Non Attacked Acc - {}" .format(acc_rgcn_non_attacked))
    # print("RGCN Attacked Acc - {}" .format(acc_rgcn_attacked))
    # print('---------------')


    # --- GCNJaccard --- #
    # 测试baseline时，modelname选择GCN、GIN
    # print('=== testing GCNJaccard ===')
    # attention = False
    # acc_jaccard_non_attacked = test(adj, defense = "GCNJaccard")
    # acc_jaccard_attacked = test(modified_adj_sparse, defense = "GCNJaccard")
    # print('---------------')
    # print("GCNJaccard Non Attacked Acc - {}" .format(acc_jaccard_non_attacked))
    # print("GCNJaccard Attacked Acc - {}" .format(acc_jaccard_attacked))
    # print('---------------')


    # --- GCNSVD --- #
    # 测试baseline时，modelname选择GCN、GIN
    # print('=== testing GCNSVD ===')
    # attention = False
    # acc_jaccard_non_attacked = test(adj, defense = "GCNSVD")
    # acc_jaccard_attacked = test(modified_adj_sparse, defense = "GCNSVD")
    # print('---------------')
    # print("GCNSVD Non Attacked Acc - {}" .format(acc_jaccard_non_attacked))
    # print("GCNSVD Attacked Acc - {}" .format(acc_jaccard_attacked))
    # print('---------------')


    # --- ProGNN --- #
    # 测试baseline时，modelname选择GCN、GIN
    # print('=== testing ProGNN ===')
    # attention = False
    # acc_jaccard_non_attacked = test(adj, defense = "ProGNN")
    # acc_jaccard_attacked = test(modified_adj_sparse, defense = "ProGNN")
    # print('---------------')
    # print("ProGNN Non Attacked Acc - {}" .format(acc_jaccard_non_attacked))
    # print("ProGNN Attacked Acc - {}" .format(acc_jaccard_attacked))
    # print('---------------')

    # --- GCORN --- #
    print('=== testing GCORN ===')
    attention = False
    acc_non_attacked,acc_attacked = test_GCORN(adj, modified_adj_sparse, defense="GCORN")
    print('---------------')
    print('For GCORN: {} - {}' .format(np.mean(acc_non_attacked) * 100,np.std(acc_non_attacked) * 100))
    print('For GCORN Peturbed : {} - {}'.format(np.mean(acc_attacked) * 100,np.std(acc_attacked) * 100))
    print('---------------')

    with open(output_file, "a") as f:
        f.write(f"{args.dataset},{args.ptb_rate},{acc_attacked}\n")

    # --- GNNGuard --- #
    # 测试baseline时，modelname选择GCN、GIN
    # print('=== testing GNNGuard ===')
    # attention = True
    # acc_non_attacked = test(adj, defense="Guard")
    # acc_attacked = test(modified_adj_sparse, defense="Guard")
    # print('---------------')
    # print("GNNGuard Non Attacked Acc - {}" .format(acc_non_attacked))
    # print("GNNGuard Attacked Acc - {}" .format(acc_attacked))
    # print('---------------')

    # --- Ours --- #
    # print('=== testing Ours ===')
    # attention = True
    # acc_non_attacked = test(adj, defense="Ours")
    # acc_attacked = test(modified_adj_sparse, defense="Ours")
    # print('---------------')
    # print("Ours Non Attacked Acc - {}" .format(acc_non_attacked))
    # print("Ours Attacked Acc - {}" .format(acc_attacked))
    # print('---------------')
