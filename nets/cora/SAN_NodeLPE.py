import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
    Graph Transformer
    
"""

# from layers.graph_transformer_layer import GraphTransformerLayer
from layers.graph_transformer_layer_no_edge_features import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class SAN_NodeLPE(nn.Module):

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

        # self.embedding_h = nn.Embedding(in_dim_node, GT_hidden_dim-LPE_dim)#Remove some embedding dimensions to make room for concatenating laplace encoding
        self.embedding_h = nn.Linear(in_dim_node, GT_hidden_dim - LPE_dim)
        self.linear_A = nn.Linear(2, LPE_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layers)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual))

        self.MLP_layer = MLPReadout(GT_out_dim, self.n_classes)


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

    def forward(self, g, h, EigVecs, EigVals, rw_probs):
        # input embedding
        h = self.embedding_h(h)
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

        # rw_probs = g.ndata["rw_probs"]
        PosEnc = self.pe_encoder(rw_probs)

        #Concatenate learned PE to input embedding
        # print("before cat: ", h.size(), PosEnc.size())
        h = torch.cat((h, PosEnc), 1)
        
        h = self.in_feat_dropout(h)
        
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
            
        # output
        h_out = self.MLP_layer(h)

        return h_out
    
    
    def loss(self, pred, label):
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(pred, label)
        loss = F.cross_entropy(pred, label)
        return loss

