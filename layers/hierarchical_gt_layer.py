import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
import nvtx
import torch_geometric

from dgl.nn import GraphConv
from layers.graph_transformer_layer_no_edge_features import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class HierarchicalGraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, num_levels, gamma, in_dim, out_dim, LPE_dim, rw_num_steps, num_heads, full_graph, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

        self.embedding_h = nn.Linear(in_dim, out_dim - LPE_dim)
        self.embedding_h_hierarchical = nn.Linear(in_dim, out_dim - LPE_dim)

        self.hierarchical_gt_layers = nn.ModuleDict()
        self.hierarchical_mlp_layers = nn.ModuleDict()
        
        self.hierarchical_gt_layers[str(0)] = GraphTransformerLayer(gamma, in_dim, out_dim, num_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual)
        self.hierarchical_mlp_layers[str(0)] = nn.Linear(in_dim, out_dim)
        for level in range(1, num_levels):
            # layers = nn.ModuleList([ GraphTransformerLayer(gamma, in_dim, out_dim, num_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(1) ]) 
            layers = GraphTransformerLayer(gamma, out_dim, out_dim, num_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual)
            self.hierarchical_gt_layers[str(level)] = layers
            self.hierarchical_mlp_layers[str(level)] = nn.Linear(out_dim, out_dim)

        self.dim_pe = LPE_dim
        pe_layers = []
        pe_layers.append(nn.Linear(rw_num_steps, 2 * self.dim_pe))
        pe_layers.append(nn.ReLU())
        for _ in range(3):
            pe_layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
            pe_layers.append(nn.ReLU())
        pe_layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
        pe_layers.append(nn.ReLU())
        self.pe_encoder = nn.Sequential(*pe_layers)

        fuse_layers = []
        fuse_layers.append(nn.Linear(2 * out_dim, out_dim))
        fuse_layers.append(nn.ReLU())
        for _ in range(3):
            fuse_layers.append(nn.Linear(out_dim, out_dim))
            fuse_layers.append(nn.ReLU())
        fuse_layers.append(nn.Linear(out_dim, out_dim))
        fuse_layers.append(nn.ReLU())
        self.fuse_mlp = nn.Sequential(*fuse_layers)
        
    def hierarchical_gt(self, g, partitions, partition_feats, partition_pe, parents_dict, children_dict):
        new_graphs = []
        for i in range(len(partitions)):
            new_level = []
            for j in range(len(partitions[i])):
                placeholder = [None] * len(partitions[i][j])
                new_level.append(placeholder)

            new_graphs.append(new_level)

        for level in range(len(partitions) - 1, 0, -1):
            for p_idx, partition in enumerate(partitions[level]):
                h_partition = []
                for g_idx, graph in enumerate(partition):
                    # probs = rw_probs[level][p_idx][g_idx]
                    probs = partition_pe[level][p_idx][g_idx]
                    if level == len(partitions) - 1:
                        # feat = h_[p_idx][g_idx]
                        feat = partition_feats[p_idx][g_idx]
                        feat = self.embedding_h(feat)
                    else:
                        feat = graph.ndata['forward_feat']
                        feat = self.embedding_h_hierarchical(feat)

                    PosEnc = self.pe_encoder(probs)
                    h = torch.cat((feat, PosEnc), 1)
                    # h = self.in_feat_dropout(h)

                    # GraphTransformer Layer
                    h = self.hierarchical_gt_layers[str(level)](graph, h)
                    h = self.hierarchical_mlp_layers[str(level)](h)

                    h_partition.append(h)
                    g1 = graph.to(g.device)
                    g1.ndata['forward_feat'] = h
                    new_graphs[level][p_idx][g_idx] = g1

                # pool the partitions to get node features for next layer
                pooled_feats = []
                for item in h_partition:
                    pooled_feats.append(torch_geometric.nn.global_mean_pool(item, batch=None)) 

                parent_level, parent_p_idx, parent_g_idx = parents_dict[(level, p_idx)]
                parents_graph = partitions[parent_level][parent_p_idx][parent_g_idx].to(g.device)
                partition_feat = torch.cat(pooled_feats)
                parents_graph.ndata['forward_feat'] = partition_feat
                new_graphs[parent_level][parent_p_idx][parent_g_idx] = parents_graph

        # do fuse on the backwards pass
        for level in range(len(partitions) - 1):
            # for p_idx, partition in enumerate(partitions[level]):
            for p_idx, partition in enumerate(new_graphs[level]):
                for g_idx, graph in enumerate(partition):
                    child_level, child_p_idx = children_dict[(level, p_idx, g_idx)]
                    for child_g_idx, child_graph in enumerate(new_graphs[child_level][child_p_idx]):
                        child_feat = child_graph.ndata['forward_feat']
                        parent_feat = graph.ndata['forward_feat'][child_g_idx].unsqueeze(0).expand(child_feat.size(0), -1)
                        fuse_input = torch.cat((child_feat, parent_feat), dim=1)
                        fuse_output = self.fuse_mlp(fuse_input)
                        child_graph.ndata['forward_feat'] = child_feat

        
        # gather in all the inputs from the last level partition
        res = torch.zeros((g.num_nodes(), self.out_channels), device=g.device)
        new_partition_feat = []
        for partition in new_graphs[-1]:
            p_feat = []
            for graph in partition:
                parent_node_ids = graph.ndata[dgl.NID]
                res[parent_node_ids] = graph.ndata['forward_feat']

                p_feat.append(graph.ndata['forward_feat'])

            new_partition_feat.append(p_feat)

        # return partition feat for next layer to use
        return res, new_partition_feat

    def forward(self, g, partitions, partition_feats, partition_pe, parents_dict, children_dict):
        h_in1 = torch.zeros((g.num_nodes(), self.out_channels), device=g.device)
        for i in range(len(partitions[-1])):
            for j in range(len(partitions[-1][i])):
                graph = partitions[-1][i][j]
                parent_node_ids = graph.ndata[dgl.NID]
                h_in1[parent_node_ids] = partition_feats[i][j]

        # h_in1 = partition_feats # for first residual connection
        
        # multi-head attention out
        h_attn_out, new_partition_feats = self.hierarchical_gt(g, partitions, partition_feats, partition_pe, parents_dict, children_dict)

        #Concat multi-head outputs
        # h = h_attn_out.view(-1, self.out_channels)
        h = h_attn_out.reshape(-1, self.out_channels)

        # h = h_in1 + h_attn_out
       
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)         

        return h, new_partition_feats
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
