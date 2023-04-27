import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
import nvtx

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func


def exp_real(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func


def exp_fake(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': L*torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias):
        super().__init__()
        
       
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.full_graph=full_graph
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            
            if self.full_graph:
                self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            
            if self.full_graph:
                self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    @nvtx.annotate("propagate_attention", color="blue")
    def propagate_attention(self, g):

        
        if self.full_graph:
            real_ids = torch.nonzero(g.edata['real']).squeeze()
            fake_ids = torch.nonzero(g.edata['real']==0).squeeze()

        else:
            real_ids = g.edges(form='eid')
            
        with nvtx.annotate("dot K_h Q_h real"):
            g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)
        
        if self.full_graph:
            with nvtx.annotate("dot K_h Q_h fake"):
                g.apply_edges(src_dot_dst('K_2h', 'Q_2h', 'score'), edges=fake_ids)
        

        # scale scores by sqrt(d)
        with nvtx.annotate("apply_edges scaling"):
            g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores for edges
        """
        with nvtx.annotate("apply_edges E matrix real "):
            g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)
        
        if self.full_graph:
            with nvtx.annotate("apply_edges E matrix fake "):
                g.apply_edges(imp_exp_attn('score', 'E_2'), edges=fake_ids)
        """
    
        if self.full_graph:
            # softmax and scaling by gamma
            L=self.gamma
            with nvtx.annotate("softmax"):
                g.apply_edges(exp_real('score', L), edges=real_ids)
                g.apply_edges(exp_fake('score', L), edges=fake_ids)
        
        else:
            g.apply_edges(exp('score'), edges=real_ids)

        # Send weighted values to target nodes
        eids = g.edges()
        with nvtx.annotate("mul V to h"):
            # g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
            g.send_and_recv(eids, fn.u_mul_e('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        with nvtx.annotate("sum score"):
            # g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))
            g.send_and_recv(eids, fn.copy_e('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))
    
    @nvtx.annotate("graph_transformer forward", color="red") 
    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        
        if self.full_graph:
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            
        V_h = self.V(h)

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        # weights_real = torch.einsum('ihd,jhd->ijh', Q_h, K_h)
        weights_real = torch.einsum('ihd,jhd->ijh', K_h, Q_h) / np.sqrt(self.out_dim)

        Q_2h = Q_2h.view(-1, self.num_heads, self.out_dim)
        K_2h = K_2h.view(-1, self.num_heads, self.out_dim)
        # weights_fake = torch.einsum('ihd,jhd->ijh', Q_2h, K_2h)
        weights_fake = torch.einsum('ihd,jhd->ijh', K_2h, Q_2h) / np.sqrt(self.out_dim)

        # real_ids = torch.nonzero(g.edata['real']).squeeze()
        # real_src, real_dst = g.find_edges(real_ids)
        real_src, real_dst = g.edges()

        weights_real_exp = torch.exp(weights_real.clamp(-5, 5))/(self.gamma + 1)
        weights_fake_exp = self.gamma * torch.exp(weights_fake.clamp(-5, 5)) / (self.gamma + 1)

        weights_fake_exp[(real_src, real_dst)] = weights_real_exp[(real_src, real_dst)]

        # set diagonal entries to 0
        vertex_range = torch.arange(0, g.number_of_nodes())
        weights_fake_exp[(vertex_range, vertex_range)] = 0
        weights_final = weights_fake_exp.view(g.number_of_nodes(), g.number_of_nodes(), -1)
        V_h_ = V_h.view(-1, self.num_heads, self.out_dim)
        out = torch.einsum('ijh,ihd->jhd', weights_final, V_h_)

        exp_sum = torch.sum(weights_final, dim=0, keepdim=True).view(g.number_of_nodes(), self.num_heads, 1)
        exp_sum_ = exp_sum + torch.full_like(exp_sum, 1e-6)
        h_out = out / exp_sum_

        """
        weights_real_exp = torch.exp(weights_real.clamp(-5, 5))/(self.gamma + 1)
        weights_fake_exp = self.gamma * torch.exp(weights_fake.clamp(-5, 5)) / (self.gamma + 1)
 
        # real_ids = torch.nonzero(g.edata['real'], as_tuple=True)
        # fake_ids = torch.nonzero(g.edata['real']==0, as_tuple=True)
        edges = g.edges()

        # print(edges.size(), weights_real_exp.size(), weights_fake_exp.size())
        weights_fake_exp[edges] = weights_real_exp[edges]

        weights = np.sqrt(self.out_dim) * weights_fake_exp
        # weights = torch.where(edges, weights_real_exp, weights_fake_exp)
        # print("Weights: ", weights.size())

        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        h_out = torch.einsum('ijh,ihd->ihd', weights, V_h)
        # print("V_h size: ", V_h.size(), self.num_heads, self.out_dim)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        
        
        if self.full_graph:
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
        
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        """
        return h_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(gamma, in_dim, out_dim//num_heads, num_heads, full_graph, use_bias)
        
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
        
    def forward(self, g, h):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        h_attn_out = self.attention(g, h)
        # print("H size: ", h_in1.size(), " output of attention: ", h_attn_out.size())
        
        #Concat multi-head outputs
        # h = h_attn_out.view(-1, self.out_channels)
        h = h_attn_out.reshape(-1, self.out_channels)
       
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

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
