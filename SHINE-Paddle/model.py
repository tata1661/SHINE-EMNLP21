import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from GCN import GCN
from utils import aggregate
paddle.set_default_dtype('float64')

class SHINE(nn.Layer):
    def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, train_idx, test_idx, params):
        super(SHINE, self).__init__()
        self.threshold = params.threshold
        self.adj = adj_dict
        self.feature = features_dict
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.type_num = len(params.type_num_node)
        self.label_num = out_features_dim[0]
        self.drop_out = params.drop_out
        self.concat_word_emb = params.concat_word_emb
        self.device = params.device
        self.train_idx = train_idx
        self.GCNS = nn.LayerList([GCN(self.in_features_dim[1], self.out_features_dim[1]), 
                                GCN(self.in_features_dim[2], self.out_features_dim[2]),
                                GCN(self.in_features_dim[3], self.out_features_dim[3])])
        self.GCNS_2 = nn.LayerList([GCN(self.out_features_dim[1], self.out_features_dim[1]), 
                                GCN(self.out_features_dim[2], self.out_features_dim[2]),
                                GCN(self.out_features_dim[3], self.out_features_dim[3])])                    
        self.refined_linear = nn.Linear(self.in_features_dim[-1], 200)
        self.final_GCN = GCN(200, self.out_features_dim[-1])
        self.final_GCN_2 = GCN(self.out_features_dim[-1], self.out_features_dim[-1])
        self.FC = nn.Linear(out_features_dim[-1], self.label_num)
        

    def embed_component(self, norm=True):
        output = []
        for i in range(self.type_num - 1):
            if i == 1 and self.concat_word_emb:
                temp_emb = paddle.concat([F.dropout(self.GCNS_2[i](self.adj[str(i + 1) + str(i + 1)],
                        F.relu(self.GCNS[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True))),
                        p=self.drop_out, training=self.training), self.feature['word_emb']], axis=-1)
                output.append(temp_emb)
            elif i == 0:
                temp_emb = F.dropout(self.GCNS_2[i](self.adj[str(i + 1) + str(i + 1)],
                        F.relu(self.GCNS[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True))),
                                p=self.drop_out, training=self.training)
                output.append(temp_emb)
            else:
                temp_emb = F.dropout(self.GCNS_2[i](self.adj[str(i + 1) + str(i + 1)],
                        F.relu(self.GCNS[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)]))),
                                p=self.drop_out, training=self.training)
                output.append(temp_emb)
        refined_text_input = aggregate(self.adj, output, self.type_num - 1)
        if norm:
            refined_text_input_normed = []
            for i in range(self.type_num - 1):
                refined_text_input_normed.append(refined_text_input[i] / (paddle.linalg.norm(refined_text_input[i],
                        p=2, axis=-1, keepdim=True) + 1e-9))
        else:
            refined_text_input_normed = refined_text_input 
        return refined_text_input_normed

    def forward(self, epoch):
        refined_text_input_normed = self.embed_component()
        Doc_features = paddle.concat(refined_text_input_normed, axis=-1)
        cos_simi_total = paddle.matmul(Doc_features, Doc_features, transpose_y=True)
        refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float64'))
        refined_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        refined_Doc_features=F.dropout(self.refined_linear(Doc_features), p=self.drop_out, training=self.training)
        final_text_output = self.final_GCN_2(refined_adj, self.final_GCN(refined_adj,refined_Doc_features))
        final_text_output=F.dropout(final_text_output,p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        return scores[self.train_idx]

    def inference(self, epoch):  
        refined_text_input_normed = self.embed_component()
        Doc_features = paddle.concat(refined_text_input_normed, axis=-1)
        refined_Doc_features=F.dropout(self.refined_linear(Doc_features), p=self.drop_out, training=self.training)
        cos_simi_total = paddle.matmul(Doc_features, Doc_features, transpose_y=True)
        refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float64'))
        refined_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        final_text_output = self.final_GCN_2(refined_adj, self.final_GCN(refined_adj,refined_Doc_features))    
        final_text_output=F.dropout(final_text_output,p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        return scores

