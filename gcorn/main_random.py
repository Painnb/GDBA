"""
Contains the complete implementation to reproduce the results of the "RANDOM"
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
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
import copy
from deeprobust.graph.global_attack import Random
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCNJaccard,GINJaccard,GCNSVD,GINSVD,RGCN,RGIN
from scipy.sparse import csr_matrix
from deeprobust.graph.defense.noisy_gcn import Noisy_GCN
from deeprobust.graph.defense.ours_gcn import Our_GCN
from deeprobust.graph.defense.ours_gin import Our_GIN
from deeprobust.graph.defense.gcn import GCN
from deeprobust.graph.defense.gin import GIN
from gcorn import GCORN
from gcorn import parseval_weight_projections, test_gcorn, train_gcorn, compute_acc_perturbation
import argparse

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'acm', 'blogcatalog', 'uai', 'flickr'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0,help='pertubation rate')
parser.add_argument('--modelname', type=str, default='GCN', choices=['Our_GCN','Our_GIN','GCN','GAT','GIN', 'JK'])
parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')
parser.add_argument('--epochs', type=int,  default=300, help='Number of epochs to train.')


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load the Dataset
data = Dataset(root='Datasets/', name=args.dataset)

def test(new_adj, defense = "GCN"):
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
        classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16,
                    nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    elif defense == "Guard":
        classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16,
                    nclass=labels.max().item() + 1, dropout=0.5, device=device)
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

    else:
        classifier = globals()[defense](nnodes=new_adj.shape[0],
                    nfeat=features.shape[1], nhid=16,
                    nclass=labels.max().item() + 1, dropout=0.5, device=device)
        attention = False

    classifier = classifier.to(device)

    classifier.fit(features, new_adj, labels, idx_train, train_iters=201,
                   idx_val=idx_val,
                   idx_test=idx_test,
                   verbose=False, attention=attention)

    classifier.eval()
    output = classifier.predict().cpu()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))

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

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

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

    # Preprocessing the data
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    adj, features = csr_matrix(adj), csr_matrix(features)

    # Setup GCN as the Surrogate Model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                dropout=0.5, with_relu=False, with_bias=False, weight_decay=5e-4,
                                                                device=device)

    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train, train_iters=201)

    attack_model = Random(model=surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
    perturbations = int(args.ptb_rate * (adj.sum()//2))
    attack_model.attack(adj, n_perturbations=perturbations, type='add')

    modified_adj = attack_model.modified_adj
    modified_adj_sparse = sp.csr_matrix(modified_adj.toarray())

    output_file = "random_gcorn.csv"
    # 初始化文件并添加表头
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("dataset,ptb_rate,acc_gcn_attacked\n")
    # print('=== testing NoisyGCN ===')
    # attention=False
    # acc_noise_clean=test_noisy(adj)
    # acc_noise_attacked=test_noisy(modified_adj)
    # print('---------------')
    # print("NoisyGCN Non Attacked Acc - {}" .format(acc_noise_clean))
    # print("NoisyGCN Attacked Acc - {}" .format(acc_noise_attacked))
    # print('---------------')

    # --- Normal GCN --- #
    # print('=== testing Normal GCN ===')
    # modified_adj = model.modified_adj
    # modified_adj = torch.FloatTensor(modified_adj.todense())
    # acc_gcn_non_attacked = test(adj)
    # acc_gcn_attacked = test(modified_adj)


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