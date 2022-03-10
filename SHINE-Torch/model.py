import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import aggregate
from GCN import GCN

class SHINE(nn.Module):
    def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, params):
        super(SHINE, self).__init__()
        self.threshold = params.threshold
        self.adj = adj_dict
        self.feature = features_dict
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.type_num = len(params.type_num_node)
        self.drop_out = params.drop_out
        self.concat_word_emb = params.concat_word_emb
        self.device = params.device
        self.GCNs = []
        self.GCNs_2 = []

        for i in range(1, self.type_num):
            self.GCNs.append(GCN(self.in_features_dim[i], self.out_features_dim[i]).to(self.device))
            self.GCNs_2.append(GCN(self.out_features_dim[i], self.out_features_dim[i]).to(self.device))            
        self.refined_linear = nn.Linear(self.out_features_dim[3]+self.out_features_dim[1]+self.out_features_dim[2] 
                            if not self.concat_word_emb else self.in_features_dim[-1], 200)
        self.final_GCN = GCN(200, self.out_features_dim[-1]).to(self.device)
        self.final_GCN_2 = GCN(self.out_features_dim[-1], self.out_features_dim[-1]).to(self.device)
        self.FC = nn.Linear(out_features_dim[-1], out_features_dim[0])

    def embed_component(self, norm=True):
        output = []
        for i in range(self.type_num - 1):
            if i == 1 and self.concat_word_emb:
                temp_emb = torch.cat([
                    F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True)),
                                p=self.drop_out, training=self.training), self.feature['word_emb']], dim=-1)
                output.append(temp_emb)     
            elif i == 0:
                temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                            self.GCNs[i](self.adj[str(i + 1) + str(i + 1)],self.feature[str(i + 1)], identity=True)),
                            p=self.drop_out, training=self.training)
                output.append(temp_emb)       
            else:
                temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                self.GCNs[i](self.adj[str(i + 1) + str(i + 1)],self.feature[str(i + 1)])),
                                p=self.drop_out, training=self.training)
                output.append(temp_emb)
        refined_text_input = aggregate(self.adj, output, self.type_num - 1) 
        if norm:
            refined_text_input_normed = []
            for i in range(self.type_num - 1):
                refined_text_input_normed.append(refined_text_input[i] / (refined_text_input[i].norm(p=2, dim=-1,keepdim=True) + 1e-9))
        else:
            refined_text_input_normed = refined_text_input
        return refined_text_input_normed
    
    def forward(self, epoch):
        refined_text_input_normed = self.embed_component()
        Doc_features = torch.cat(refined_text_input_normed, dim=-1)
        refined_text_input_after_final_linear=F.dropout(self.refined_linear(Doc_features), 
                                                        p=self.drop_out, training=self.training)
        cos_simi_total = torch.matmul(Doc_features, Doc_features.t())
        refined_adj_tmp = cos_simi_total * (cos_simi_total > self.threshold).float()
        refined_adj = refined_adj_tmp / (refined_adj_tmp.sum(dim=-1, keepdim=True) + 1e-9)
        final_text_output = self.final_GCN_2(refined_adj, 
                                self.final_GCN(refined_adj,refined_text_input_after_final_linear))
        final_text_output = F.dropout(final_text_output, p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        return scores



