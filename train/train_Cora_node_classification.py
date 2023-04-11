"""
    Utility function for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy

def train_epoch(model, optimizer, device, g, epoch, LPE):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0

    train_mask, val_mask = g.ndata["train_mask"], g.ndata["val_mask"]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    optimizer.zero_grad()

    batch_graphs = g.to(device)
    batch_x = features.to(device)
    # batch_e = g.edata["feat"].flatten().long().to(device)
    batch_labels = labels.to(device)

    # batch_graphs = batch_graphs.to(device)
    # batch_x = batch_graphs.ndata['feat'].to(device)
    # batch_e = batch_graphs.edata['feat'].flatten().long().to(device)

    # batch_labels = batch_labels.to(device)
    # optimizer.zero_grad()  
    
    batch_EigVecs = g.ndata['EigVecs'].to(device)
    #random sign flipping
    sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
    sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
    
    batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
    print("DIMS: ", features.size(), labels.size())
    batch_scores = model.forward(batch_graphs, batch_x, batch_EigVecs, batch_EigVals)

    loss = model.loss(batch_scores[train_mask], batch_labels[train_mask])
    loss.backward()
    optimizer.step()
    epoch_loss += loss.detach().item()
    epoch_train_acc += accuracy(batch_scores[train_mask], batch_labels[train_mask])

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network(model, device, g, epoch, LPE, val_or_test="test"):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0

    train_mask, val_mask = g.ndata["train_mask"], g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    with torch.no_grad():
        train_mask, val_mask = g.ndata["train_mask"], g.ndata["val_mask"]
        features = g.ndata["feat"]
        labels = g.ndata["label"]
        optimizer.zero_grad()

        batch_graphs = g.to(device)
        batch_x = features.to(device)
        batch_labels = labels.to(device)

        if LPE == 'node':
            batch_EigVecs = g.ndata['EigVecs'].to(device)
            batch_EigVals = g.ndata['EigVals'].to(device)
            batch_scores = model.forward(g, x, batch_EigVecs, batch_EigVals)
                
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


