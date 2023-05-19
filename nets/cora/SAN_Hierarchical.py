import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np
import torch_geometric

"""
    Graph Transformer
    
"""

from layers.graph_transformer_layer_no_edge_features import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class SAN_Hierarchical(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        self.n_classes = net_params['n_classes']
        
        full_graph = net_params['full_graph']
        gamma = net_params['gamma']
        
        LPE_layers = net_params['LPE_layers']
        LPE_dim = net_params['LPE_dim']
        LPE_n_heads = net_params['LPE_n_heads']
        
        GT_layers = net_params['GT_layers']
        GT_hidden_dim = net_params['GT_hidden_dim']
        GT_out_dim = net_params['GT_out_dim']
        GT_n_heads = net_params['GT_n_heads']
        
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.hidden_dim = GT_hidden_dim
        self.out_dim = GT_out_dim

        # self.embedding_h = nn.Embedding(in_dim_node, GT_hidden_dim-LPE_dim)#Remove some embedding dimensions to make room for concatenating laplace encoding
        self.embedding_h = nn.Linear(in_dim_node, GT_hidden_dim - LPE_dim)

        self.embedding_h_test = nn.Linear(in_dim_node, GT_hidden_dim)

        self.linear_A = nn.Linear(2, LPE_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layers)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual))

        self.MLP_layer = MLPReadout(GT_out_dim, self.n_classes)

        self.MLP_layer_hierarchical = MLPReadout(GT_out_dim, GT_hidden_dim)

        self.embedding_h_hierarchical = nn.Linear(GT_hidden_dim, GT_hidden_dim)


        # Using a different type of encoder, kernel_pos_encoder
        self.dim_in = in_dim_node
        self.dim_pe = LPE_dim
        self.num_rw_steps = 5
        self.n_layers = LPE_layers
        self.linear_x = nn.Linear(self.dim_in, GT_hidden_dim - self.dim_pe)

        pe_layers = []
        pe_layers.append(nn.Linear(self.num_rw_steps, 2 * self.dim_pe))
        pe_layers.append(nn.ReLU())
        for _ in range(self.n_layers - 2):
            pe_layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
            pe_layers.append(nn.ReLU())

        pe_layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
        pe_layers.append(nn.ReLU())
        self.pe_encoder = nn.Sequential(*pe_layers)

        fuse_layers = []
        fuse_layers.append(nn.Linear(2 * GT_hidden_dim, GT_hidden_dim))
        fuse_layers.append(nn.ReLU())
        for _ in range(self.n_layers - 1):
            fuse_layers.append(nn.Linear(GT_hidden_dim, GT_hidden_dim))
            fuse_layers.append(nn.ReLU())

        self.fuse_mlp = nn.Sequential(*fuse_layers)

    def compute_partition(self, partitions):
        res = []
        for graph in partitions:
            h = graph.ndata['feat']
            h = self.embedding_h(h)
            rw_probs = g.ndata["rw_probs"]
            # print("rw_probs: ", type(rw_probs))
            PosEnc = self.pe_encoder(rw_probs)
            h = torch.cat((h, PosEnc), 1)
            h = self.in_feat_dropout(h)
            # GraphTransformer Layers
            for conv in self.layers:
                h = conv(graph, h)
                h_out = self.MLP_layer(h)
                cluster_repr = torch_geometric.nn.global_mean_pool(h_out)
            res.append((h_out, cluster_repr))
        return res

    def forward(self, g, h_, EigVecs, EigVals, rw_probs, partitions, parents_dict, children_dict):
        # partitions is a list of lists, length is num_levels
        # input embedding

        # h = self.embedding_h(h)
        # TODO: currently just 1 level

        """
        # e = self.embedding_e(e) 
          
        PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float() # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(PosEnc) # (Num nodes) x (Num Eigenvectors) x 2
        
        PosEnc[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
        PosEnc = torch.transpose(PosEnc, 0 ,1).float() # (Num Eigenvectors) x (Num nodes) x 2
        PosEnc = self.linear_A(PosEnc) # (Num Eigenvectors) x (Num nodes) x PE_dim
        
        
        #1st Transformer: Learned PE
        PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:,:,0]) 

        #remove masked sequences
        PosEnc[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan') 
        
        #Sum pooling
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False)
        """
        partition_feat = []
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
                    # rw_probs = graph.ndata["rw_probs"]
                    probs = rw_probs[level][p_idx][g_idx]
                    if level == len(partitions) - 1:
                        feat = h_[p_idx][g_idx]
                        # print("feat size: ", feat.size())
                        feat = self.embedding_h(feat)
                    else:
                        feat = graph.ndata['feat']
                        feat = self.embedding_h_hierarchical(feat)

                    # print("probs shape: ", probs.size())
                    PosEnc = self.pe_encoder(probs)
                    # feat = graph.ndata['feat']
                    h = torch.cat((feat, PosEnc), 1)
                    h = self.in_feat_dropout(h)

                    # GraphTransformer Layers
                    for conv in self.layers:
                        h = conv(graph, h)
                    
                    h = self.MLP_layer_hierarchical(h)
                    h_partition.append(h)
                    g1 = graph.to(self.device)
                    g1.ndata['feat'] = h
                    # partitions[level][p_idx][g_idx] = g1
                    new_graphs[level][p_idx][g_idx] = g1

                # pool the partitions to get node features for next layer
                pooled_feats = []
                for item in h_partition:
                    pooled_feats.append(torch_geometric.nn.global_mean_pool(item, batch=None)) 

                parent_level, parent_p_idx, parent_g_idx = parents_dict[(level, p_idx)]
                
                parents_graph = partitions[parent_level][parent_p_idx][parent_g_idx].to(self.device)
                partition_feat = torch.cat(pooled_feats)
                parents_graph.ndata['feat'] = partition_feat
                # partitions[parent_level][parent_p_idx][parent_g_idx] = parents_graph
                new_graphs[parent_level][parent_p_idx][parent_g_idx] = parents_graph

        # do fuse
        for level in range(len(partitions) - 1):
            # for p_idx, partition in enumerate(partitions[level]):
            for p_idx, partition in enumerate(new_graphs[level]):
                for g_idx, graph in enumerate(partition):
                    child_level, child_p_idx = children_dict[(level, p_idx, g_idx)]
                    # for child_g_idx, child_graph in enumerate(partitions[child_level][child_p_idx]):
                    for child_g_idx, child_graph in enumerate(new_graphs[child_level][child_p_idx]):
                        assert(child_graph is not None)
                        child_feat = child_graph.ndata['feat']
                        parent_feat = graph.ndata['feat'][child_g_idx].unsqueeze(0).expand(child_feat.size(0), -1)
                        fuse_input = torch.cat((child_feat, parent_feat), dim=1)
                        fuse_output = self.fuse_mlp(fuse_input)
                        
                        child_graph.ndata['feat'] = child_feat

        
        # print("g number of nodes: ", g.num_nodes())
        # gather in all the inputs from the last level partition
        res = torch.zeros((g.num_nodes(), self.hidden_dim), device=EigVecs.device)
        for partition in new_graphs[-1]:
            for graph in partition:
                parent_node_ids = graph.ndata[dgl.NID]
                # g.ndata['feat'][parent_node_ids] = graph.ndata['feat']
                res[parent_node_ids] = graph.ndata['feat']

        # print("res nonzero: ", torch.count_nonzero(res))
        return res

        # h_test = self.embedding_h_test(h)
        # return h_test
        return h

    
    
    def loss(self, pred, label):
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(pred, label)
        loss = F.cross_entropy(pred, label)
        return loss

