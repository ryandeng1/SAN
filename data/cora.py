
import time
import os
import pickle
import numpy as np

import dgl
import torch
import torch.nn.functional as F

from scipy import sparse as sp
import numpy as np
import networkx as nx

import hashlib
from dgl.data import CoraGraphDataset


class CoraDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, g, mask_key):
        feat = g.ndata['feat']
        ndata = g.ndata[mask_key]
        label = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

        label = g.ndata['label']

        self.node_labels = [label]
        self.graph_lists = [g]
        self.n_samples = 1

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.node_labels[idx]

class CoraDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset = CoraGraphDataset()
        self.dataset = dataset
        self.g = dataset[0]
        self.name = "CORA"

    def _laplace_decomp(self, max_freqs):
        self.g = laplace_decomp(self.g, max_freqs)

    def _make_full_graph(self):
        self.g = make_full_graph(self.g)

    def _add_edge_laplace_feats(self):
        self.g = add_edge_laplace_feats(self.g)

def laplace_decomp(g, max_freqs):
    # Laplacian
    n = g.number_of_nodes()
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
    else:
        g.ndata['EigVecs']= EigVecs
        
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    
    #Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
    print("LAPLACE DECOMP: ", g.ndata.keys())
    return g



def make_full_graph(g):

    
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    #Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']
    for k in g.ndata:
        full_g.ndata[k] = g.ndata[k]

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass
    
    #Populate edge features w/ 0s
    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
    #Copy real edge data over
    # full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    # full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    
    print("EDGES: ", full_g.edges(), full_g.edata)
    return full_g


def add_edge_laplace_feats(g):

    
    EigVals = g.ndata['EigVals'][0].flatten()
    
    source, dest = g.find_edges(g.edges(form='eid'))
    
    #Compute diffusion distances and Green function
    g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs']-g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(),1).unsqueeze(2)
    
    #No longer need EigVecs and EigVals stored as node features
    del g.ndata['EigVecs']
    del g.ndata['EigVals']
    
    return g




    
class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/SBMs/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
                
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):

        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels
    

    def _laplace_decomp(self, max_freqs):
        self.train.graph_lists = [laplace_decomp(g, max_freqs) for g in self.train.graph_lists]
        self.val.graph_lists = [laplace_decomp(g, max_freqs) for g in self.val.graph_lists]
        self.test.graph_lists = [laplace_decomp(g, max_freqs) for g in self.test.graph_lists]
    

    def _make_full_graph(self):
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]


    def _add_edge_laplace_feats(self):
        self.train.graph_lists = [add_edge_laplace_feats(g) for g in self.train.graph_lists]
        self.val.graph_lists = [add_edge_laplace_feats(g) for g in self.val.graph_lists]
        self.test.graph_lists = [add_edge_laplace_feats(g) for g in self.test.graph_lists]  
