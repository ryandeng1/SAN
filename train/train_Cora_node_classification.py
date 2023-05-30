"""
    Utility function for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_Cora as accuracy

def train_epoch(model, optimizer, device, g, epoch, LPE, partitions=[], parents_dict={}, children_dict={}, eigvecs=None, eigvals=None):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0

    train_mask = g.ndata["train_mask"]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    optimizer.zero_grad()

    batch_graphs = g.to(device)
    batch_x = features.to(device)
    # batch_e = g.edata["feat"].flatten().long().to(device)
    batch_labels = labels.to(device)

    if eigvecs is None:
        batch_EigVecs = g.ndata['EigVecs'].to(device)
    else:
        batch_EigVecs = eigvecs.to(device)

    #random sign flipping
    sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
    sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
    
    if eigvecs is None:
        batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
    else:
        batch_EigVals = eigvals.to(device)
    batch_EigVecs = batch_EigVecs * sign_flip.unsqueeze(0)
    rw_probs = g.ndata['rw_probs'].to(device)

    """
    feat = []
    for partition in partitions[-1]:
        feat_partition = []
        for graph in partition:
            feat_partition.append(graph.ndata['feat'].to(device))
        feat.append(feat_partition)

    rw_probs = []
    for level in range(len(partitions)):
        rw_probs_level = []
        for partition in partitions[level]:
            rw_probs_partition = []
            for graph in partition:
                rw_probs_partition.append(graph.ndata['rw_probs'].to(device))
            rw_probs_level.append(rw_probs_partition)
        rw_probs.append(rw_probs_level)
    """

    # batch_scores = model.forward(batch_graphs, feat, batch_EigVecs, batch_EigVals, rw_probs, partitions, parents_dict, children_dict)
    batch_scores = model.forward(batch_graphs, batch_x, batch_EigVecs, batch_EigVals, rw_probs)
    # batch_scores = model.forward(batch_graphs, batch_x)

    loss = model.loss(batch_scores[train_mask], batch_labels[train_mask])
    loss.backward()
    optimizer.step()
    epoch_loss += loss.detach().item()

    _, indices = torch.max(batch_scores[train_mask], dim=1)
    correct = torch.sum(indices == batch_labels[train_mask])
    train_acc = correct.item() * 1.0 / len(batch_labels[train_mask])
    epoch_train_acc += train_acc

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, g, epoch, LPE, partitions=[], parents_dict={}, children_dict={}, val_or_test="test", eigvecs=None, eigvals=None):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0

    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    with torch.no_grad():
        train_mask, val_mask = g.ndata["train_mask"], g.ndata["val_mask"]
        features = g.ndata["feat"]
        labels = g.ndata["label"]

        batch_graphs = g.to(device)
        batch_x = features.to(device)
        batch_labels = labels.to(device)

        if LPE == 'node':
            if eigvecs is None:
                batch_EigVecs = g.ndata['EigVecs'].to(device)
                batch_EigVals = g.ndata['EigVals'].to(device)
            else:
                batch_EigVecs = eigvecs.to(device)
                batch_EigVals = eigvals.to(device)
            rw_probs = g.ndata['rw_probs'].to(device)
            """
            rw_probs = []
            for level in range(len(partitions)):
                rw_probs_level = []
                for partition in partitions[level]:
                    rw_probs_partition = []
                    for graph in partition:
                        rw_probs_partition.append(graph.ndata['rw_probs'].to(device))
                    rw_probs_level.append(rw_probs_partition)
                rw_probs.append(rw_probs_level)
            feat = []
            for partition in partitions[-1]:
                feat_partition = []
                for graph in partition:
                    feat_partition.append(graph.ndata['feat'].to(device))
                feat.append(feat_partition)
            """
            # batch_scores = model.forward(batch_graphs, feat, batch_EigVecs, batch_EigVals, rw_probs, partitions, parents_dict, children_dict)
            batch_scores = model.forward(batch_graphs, batch_x, batch_EigVecs, batch_EigVals, rw_probs)
            # batch_scores = model.forward(batch_graphs, batch_x)
                
            if val_or_test == "test":
                loss = model.loss(batch_scores[test_mask], batch_labels[test_mask])
                epoch_test_loss += loss.detach().item()
                epoch_test_acc += accuracy(batch_scores[test_mask], batch_labels[test_mask])
            else:
                loss = model.loss(batch_scores[val_mask], batch_labels[val_mask])
                epoch_test_loss += loss.detach().item()
                epoch_test_acc += accuracy(batch_scores[val_mask], batch_labels[val_mask])
        else:
            assert(False)
        
    return epoch_test_loss, epoch_test_acc


