import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty([in_features, out_features], dtype=torch.float),
            requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty([out_features], dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, inputs, identity=False): 
        if identity:
            return torch.matmul(adj,self.weight)
        return torch.matmul(adj, torch.matmul(inputs, self.weight))
