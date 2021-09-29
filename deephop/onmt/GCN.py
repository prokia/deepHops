import dgl
import dgl.function as fn
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

def gcn_reduce(nodes):
    msgs = torch.cat((nodes.mailbox['h'], nodes.data['h'].unsqueeze(1)), dim = 1)
    msgs = torch.mean(msgs, dim = 1)
    return {'h': msgs}

def gcn_msg(edges):

    return {'h': edges.src['h']}



class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(NodeApplyModule, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias = True)
    

    def forward(self, node):
        h = self.fc(node.data['h'])
        h = F.relu(h)
        return {'h' : h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats)

    def forward(self, g, features):
        g.ndata['h'] = features

        g.update_all(gcn_msg, gcn_reduce)

        g.apply_nodes(func = self.apply_mod)
     

        return g.ndata.pop('h')


# 2 layers GCN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(256, 256)
        self.gcn2 = GCN(256, 256)
        # self.fc = nn.Linear(70, 15)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        g.ndata['h'] = x     
        # hg = dgl.mean_nodes(g, 'h')
        return x
