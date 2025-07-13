import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # GraphConvolution forward。input*weight
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout_rate):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout_rate

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj

class GCN(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, hidden3, z_emb_size, dropout_rate):#初始化
        super(GCN, self).__init__()
        self.dropout = dropout_rate

        self.gc1 = GraphConvolution(in_dim, hidden1)
        self.gc2 = GraphConvolution(hidden1, hidden2)
        self.gc3 = GraphConvolution(hidden2, hidden3)
        self.gc4 = GraphConvolution(hidden3, z_emb_size)
        self.dc = InnerProductDecoder(dropout_rate)

    def gcn_encoder(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x3 = F.relu(self.gc3(x2, adj))
        x3 = F.dropout(x3, self.dropout, training=self.training)

        emb = F.relu(self.gc4(x3, adj))

        return emb

    def gcn_decoder(self, emb):
        adj_hat = self.dc(emb)
        return adj_hat

    def forward(self, x, adj):
        emb = self.gcn_encoder(x, adj)
        adj_hat = self.gcn_decoder(emb)
        return emb, adj_hat