import math
import paddle
import paddle.nn as nn

class GCN(nn.Layer):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        stdv = 1. / math.sqrt(self.out_features)

        self.weight = self.create_parameter(shape=[self.in_features, self.out_features], dtype='float64', 
                        default_initializer=nn.initializer.Uniform(low=-stdv, high=stdv))
        self.add_parameter('weight', self.weight)

        if bias:
            self.bias = paddle.create_parameter(shape=[self.out_features], dtype='float64',
                        default_initializer=nn.initializer.Uniform(low=-stdv, high=stdv))
            self.add_parameter('bias', self.bias)
        else:
            self.add_parameter('bias', None)

    def forward(self, adj, inputs, identity=False):
        if identity:
            return paddle.matmul(adj, self.weight)
        return paddle.matmul(adj, paddle.matmul(inputs, self.weight))

