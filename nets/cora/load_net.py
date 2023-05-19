"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.cora.SAN_NodeLPE import SAN_NodeLPE
from nets.cora.SAN_EdgeLPE import SAN_EdgeLPE
from nets.cora.SAN import SAN
from nets.cora.SAN_Hierarchical import SAN_Hierarchical

def NodeLPE(net_params):
    return SAN_NodeLPE(net_params)

def EdgeLPE(net_params):
    return SAN_EdgeLPE(net_params)

def NoLPE(net_params):
    return SAN(net_params)

def Hierarchical(net_params):
    return SAN_Hierarchical(net_params)

def gnn_model(LPE, net_params):
    model = {
        'edge': EdgeLPE,
        'node': NodeLPE,
        'hierarchical': Hierarchical,
        'none': NoLPE
    }
        
    return model[LPE](net_params)
