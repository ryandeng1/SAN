
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
        a = self.g.adjacency_matrix().to_dense()

    def _laplace_decomp(self, max_freqs):
        self.g = laplace_decomp(self.g, max_freqs)

    def _make_full_graph(self):
        self.g = make_full_graph(self.g)

    def _add_edge_laplace_feats(self):
        self.g = add_edge_laplace_feats(self.g)

    def create_random_projection_partition(self):
        self.partitions = random_projection_partition(self.g)

    def get_rw_probs(self, num_steps):
        # edges = self.g.edges()
        # rw_probs = get_rw_landing_probs(num_steps, edges, None, self.g.num_nodes())
        # self.g.ndata['rw_probs'] = rw_probs 
        # self.g = rw_probs(self.g, num_steps, edges, None, self.g.num_nodes(), 0)
        rw_probs = dgl.random_walk_pe(self.g, num_steps)
        self.g.ndata['rw_probs'] = rw_probs

def rw_probs(g, num_steps, edges, edge_weight=None, num_nodes=None, space_dim=0):
    assert(False, "For cora, should be calling dataset.get_rw_probs() instead of this function")
    return g
    rw_probs = get_rw_landing_probs(num_steps, edges, edge_weight, num_nodes, space_dim)
    g.ndata['rw_probs'] = rw_probs
    return g


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
    return g


def make_full_graph(g):
    return g

    """
    
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
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.number_of_edges(), dtype=torch.long)
    return full_g
    """


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

# return only the bottom level of partitions, need to invoke this method multiple times
def random_projection_partition(num_levels, g):
    graphs = [[g]]
    node_cluster_info = []
    for i in range(num_levels - 1):
        res = []
        for j in range(len(graphs[i])):
            graph = graphs[i][j]
            EigVecs = graph.ndata['EigVecs']
            EigVals = graph.ndata['EigVals']
            rand_vec = torch.rand((EigVecs.size(1)), device=EigVecs.device)
            rand_vec = EigVecs @ rand_vec
            pos_nodes = (rand_vec > 0)
            neg_nodes = (rand_vec <= 0)
            # if i == 0, then graph is g, so look at its nodes rather than the parent nodes
            if i != 0:
                pos_nodes = graph.ndata[dgl.NID][pos_nodes]
                neg_nodes = graph.ndata[dgl.NID][neg_nodes]

            pos_nodes_graph = g.subgraph(pos_nodes)
            neg_nodes_graph = g.subgraph(neg_nodes)
            res.extend([pos_nodes_graph, neg_nodes_graph])
        graphs.append(res)

    # technically only need to look at the "bottom" level
    last_level_graphs = [graph for graph in graphs[-1] if graph.number_of_nodes() > 0]
    for idx, graph in enumerate(last_level_graphs):
        for node in graph.ndata[dgl.NID]:
            node_cluster_info.append((node.item(), idx))

    # return lowest level of hierarchy as this acts as one "level" in the grand scheme of things
    # return graphs[-1], node_cluster_info
    return last_level_graphs, node_cluster_info

def get_coarsening_matrix(g, partitions, node_cluster_info, device):
    coarsen_operator = torch.zeros((g.number_of_nodes(), len(partitions)), device=device)
    idx = torch.tensor(node_cluster_info, device=device)
    coarsen_operator[idx[:, 0], idx[:, 1]] = 1

    node_to_cluster = {}
    for node, cluster_idx in node_cluster_info:
        node_to_cluster[node] = cluster_idx

    adj = g.adj().to(device)
    intermediate = torch.sparse.mm(adj, coarsen_operator)
    coarsen_matrix = torch.mm(torch.transpose(coarsen_operator, 0, 1), intermediate)
    coarsen_matrix = (coarsen_matrix > 0).float()

    # test if matrix is valid
    start_tensor, end_tensor = g.edges()
    start_list = start_tensor.tolist()
    end_list = end_tensor.tolist()
    for start, end in zip(start_list, end_list):
        cluster_start = int(coarsen_operator[start][node_to_cluster[start]].item())
        cluster_end = int(coarsen_operator[start][node_to_cluster[end]].item())
        if cluster_start != cluster_end:
            assert(coarsen_matrix[cluster_start][cluster_end].item() == 1)

    # zero out diagonal entries
    return coarsen_matrix - torch.eye(len(partitions), device=device)
    # return coarsen_matrix

# return the hierarchy along with node_cluster_info
def get_hierarchy(g, num_split, num_levels, device):
    res_graphs = [[[g]]]
    res_node_cluster_info = [[None]]
    parents = {}
    children = {}
    res_coarse_graphs = []
    for i in range(1, num_levels):
        curr_level_graphs = []
        curr_level_node_cluster_info = []
        curr_level_coarse_graphs = []
        start = res_graphs[i - 1]
        for partition_idx, partition in enumerate(start):
            coarse_graphs_per_partition = []
            for graph_idx, graph in enumerate(partition):
                partitions, node_cluster_info = random_projection_partition(num_split, graph)
                curr_level_graphs.append(partitions)
                curr_level_node_cluster_info.append(node_cluster_info)
                parents[(i, len(curr_level_graphs) - 1)] = (i - 1, partition_idx, graph_idx)
                children[(i - 1, partition_idx, graph_idx)] = (i, len(curr_level_graphs) - 1)


                # parents[(num_levels - i - 1, len(curr_level_graphs) - 1)] = (i + 1, partition_idx, graph_idx)

                coarsening_matrix = get_coarsening_matrix(graph, partitions, node_cluster_info, device)
                tmp = torch.nonzero(coarsening_matrix)
                src, dst = tmp[:, 0], tmp[:, 1]
                # print("check size: ", src.size(), dst.size())
                # src, dst = torch.nonzero(coarsening_matrix).cpu().detach().numpy()
                coarse_graph = dgl.graph((src, dst))
                # coarse_graph = dgl.graph(torch.nonzero(coarsening_matrix))
                # coarse_graphs.append(coarse_graph)
                coarse_graphs_per_partition.append(coarse_graph)

            curr_level_coarse_graphs.append(coarse_graphs_per_partition)

        res_coarse_graphs.append(curr_level_coarse_graphs) 
        res_graphs.append(curr_level_graphs)
        res_node_cluster_info.append(curr_level_node_cluster_info)

    return res_graphs, res_node_cluster_info, res_coarse_graphs, parents, children

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    assert(False, "this function should not be called. Rw-probs, should use dgl's built-in function to obtain these positional encodings")
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if type(ksteps) != list:
        ksteps = [ksteps]
    source, dest = edge_index[0], edge_index[1]
    edge_tensor = torch.stack((source, dest)).to(source.device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_tensor.size(1), device=source.device)

    assert(num_nodes is not None)
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_tensor.numel() == 0:
        # P = edge_index.new_zeros((1, num_nodes, num_nodes))
        P = edge_tensor.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        # P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
        P = torch.diag(deg_inv) @ to_dense_adj(edge_tensor, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)
    return rw_landing
