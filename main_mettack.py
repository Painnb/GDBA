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
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import *
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset
import argparse
from scipy.sparse import csr_matrix
import pickle
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import normalize
import scipy
import numpy as np
from sklearn.preprocessing import normalize
import torch_geometric
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from deeprobust.graph.defense.gcn import GCN
from deeprobust.graph.defense.gin import GIN
from deeprobust.graph.defense.gin_preprocess import GINJaccard, GINSVD
from deeprobust.graph.defense.r_gin import RGIN
from deeprobust.graph.defense.noisy_gcn import Noisy_GCN
from deeprobust.graph.defense.noisy_gin import Noisy_GIN
from deeprobust.graph.defense.migcnppr import MIGCN
from deeprobust.graph.defense.miginppr import MIGIN
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='blogcatalog', choices=['cora', 'citeseer', 'acm', 'blogcatalog', 'uai', 'flickr'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2, help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')
parser.add_argument('--modelname', type=str, default='MIGCN', choices=['MIGCN','MIGIN','GCN','GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJacc', choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=False, choices=[True, False])

parser.add_argument('--beta_max', type=float, default=0.15)
parser.add_argument('--beta_min', type=float, default=0.01)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: %s' % device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)
torch.set_num_threads(1)

# Load the Dataset
data = Dataset(root='Datasets/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels

# Extract the Train/Val/Test idx
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
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

    elif defense == "MIGCN":
        if args.modelname == "MIGCN":
            classifier = MIGCN(nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        elif args.modelname == "MIGIN":
            classifier = MIGIN(nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False
    elif defense == "MIGCN":
        if args.modelname == "MIGIN":
            classifier = MIGCN(nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        elif args.modelname == "MIGIN":
            classifier = MIGIN(nfeat=features.shape[1],
                nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False
    else:
        classifier = globals()[defense](nnodes=adj.shape[0], nhid=16,
                        nfeat=features.shape[1], nclass=labels.max().item() + 1,
                                                    dropout=0.5, device=device)
        attention = False

    classifier = classifier.to(device)

    classifier.fit(features, adj, labels, idx_train, train_iters=201,idx_val=idx_val, idx_test=idx_test, verbose=False, attention=attention)
    classifier.eval()

    acc_test = classifier.test(idx_test)
    if isinstance(acc_test, tuple):
        acc_test = acc_test[0]

    return acc_test



if __name__ == '__main__':
    """
    Main function containing the Mettack implementation, please note that you
    need to uncomment the last part to use the other benchamarks
    """
    output_file = "results_gcn_mettack.csv"
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
    # print("Noisy_GCN Non Attacked Acc - {}".format(acc_noise_clean))
    # print("Noisy_GCN Attacked Acc - {}" .format(acc_noise_attacked))
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


    # --- GDBA --- #
    print('=== testing  GDBA ===')
    # acc_gcn_non_attacked = test(adj)
    acc_gcn_attacked = test(modified_adj_sparse)
    print('---------------')
    # print("MIGCN Non Attacked Acc - {}" .format(acc_gcn_non_attacked))
    print("MIGCN Attacked Acc - {}" .format(acc_gcn_attacked))
    print('---------------')
    with open(output_file, "a") as f:
        f.write(f"{args.dataset},{args.ptb_rate},{acc_gcn_attacked}\n")
