"""
Utils script containing utils function to be used
"""

import torch
import torch.nn.functional as F



def train_gcorn(model_local, optimizer_local, data_local, adj_normalized):
    """
    Function to train a model.
    ---
    Input:
        model_local: Instance of the model to be trained
        optimizer_local: The optimizer related to the model
        data_local: The data on which to be trained
        adj_normalized: The (Normalized) adjacency to be used.
    """
    model_local.train()
    optimizer_local.zero_grad()
    out = model_local(data_local.features, adj_normalized)
    if data_local.labels.dtype != torch.long:
        data_local.labels = torch.from_numpy(data_local.labels).long().cuda()
    loss = F.cross_entropy(out[data_local.train_mask],data_local.labels[data_local.train_mask])
    loss.backward()
    optimizer_local.step()
    return model_local, float(loss)

def parseval_train(model_local, optimizer_local, data_local, adj_normalized,
                                                        retraction_par = 0.001):
    """
    Function to train a parseval model. The main difference is the usage of the
    parseval regularization and projection.
    ---
    Input:
        model_local: Instance of the model to be trained
        optimizer_local: The optimizer related to the model
        data_local: The data on which to be trained
        adj_normalized: The (Normalized) adjacency to be used.
    """
    model_local.train()
    optimizer_local.zero_grad()
    out = model_local(data_local.features, adj_normalized)
    loss = F.cross_entropy(out[data_local.train_mask], data_local.labels[data_local.train_mask])
    loss.backward()
    optimizer_local.step()
    from parseval_constraint import parseval_weight_projections
    model_local = parseval_weight_projections(model_local, retraction_par)
    return model_local, float(loss)


@torch.no_grad()
def test_gcorn(model_local, data_local, adj_normalized):
    """
    Function to test a model.
    ---
    Input:
        model_local: Instance of the model to be trained
        optimizer_local: The optimizer related to the model
        data_local: The data on which to be trained
        adj_normalized: The (Normalized) adjacency to be used.
    """
    model_local.eval()
    pred = out = model_local(data_local.features, adj_normalized).argmax(dim=-1)

    accs = []
    for mask in [data_local.train_mask, data_local.val_mask, data_local.test_mask]:
        accs.append(int((pred[mask] == data_local.labels[mask]).sum()) / int(mask.sum()))
    return accs

# 源代码是特征攻击，扰动的是x
# def compute_acc_perturbation(model_local, data_local, adj_local, x_pertubed_local):
#     """
#     Function to test a model in the clean and attacked setting in the case of
#     node-feature based adversarial attacks.
#     ---
#     Input:
#         model_local: Instance of the model to be trained
#         data_local: The data on which to be trained
#         adj_local: The (Normalized) adjacency to be used.
#         x_pertubed_local: The perturbed/attacked node features.

#     """
#     model_local.eval()
#     out_1 = model_local(data_local.x, adj_local)
#     pred_1 = out_1.argmax(dim=-1)
#     acc_1 = int((pred_1[data_local.test_mask] == data_local.y[data_local.test_mask]).sum()) / int(data_local.test_mask.sum())

#     out_2 = model_local(x_pertubed_local, adj_local)
#     pred_2 = out_2.argmax(dim=-1)
#     acc_2 = int((pred_2[data_local.test_mask] == data_local.y[data_local.test_mask]).sum()) / int(data_local.test_mask.sum())

#     return acc_1, acc_2, out_1, out_2

# 我们改成结构攻击，扰动的是adj
def compute_acc_perturbation(model_local, data_local, adj_original, adj_perturbed):
    """
    Function to test a model in the clean and attacked setting in the case of
    node-feature based adversarial attacks.
    ---
    Input:
        model_local: Instance of the model to be trained
        data_local: The data on which to be trained
        adj_original: The (Normalized) original adjacency matrix.
        adj_perturbed: The perturbed/attacked adjacency matrix.

    """
    model_local.eval()

    # Test with original adjacency matrix
    out_original = model_local(data_local.features, adj_original)
    pred_original = out_original.argmax(dim=-1)
    acc_original = int((pred_original[data_local.test_mask] == data_local.labels[data_local.test_mask]).sum()) / int(data_local.test_mask.sum())

    # Test with perturbed adjacency matrix
    out_perturbed = model_local(data_local.features, adj_perturbed)
    pred_perturbed = out_perturbed.argmax(dim=-1)
    acc_perturbed = int((pred_perturbed[data_local.test_mask] == data_local.labels[data_local.test_mask]).sum()) / int(data_local.test_mask.sum())

    return acc_original, acc_perturbed, out_original, out_perturbed

if __name__ == "__main__":
    pass