import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN import GCN
from utils import fetch_adj, fetch_feature

class SHINE(nn.Module):
    def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, params):
        super(SHINE, self).__init__()
        self.threshold = params.threshold
        self.adj_dict = adj_dict
        self.features_dict = features_dict
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.type_num = len(params.type_num_node)
        self.drop_out = params.drop_out
        self.concat_word_emb = params.concat_word_emb
        self.device = params.device
        self.bilinears = nn.ModuleList()
        self.GCNs = []
        self.GCNs_2 = []

        for i in range(1, self.type_num):
            self.GCNs.append(GCN(self.in_features_dim[i], self.out_features_dim[i]).to(self.device))
            self.GCNs_2.append(GCN(self.out_features_dim[i], self.out_features_dim[i]).to(self.device))
            self.bilinears.append(nn.Linear(self.out_features_dim[i], self.out_features_dim[i], bias=False))
            
        self.refined_linear = nn.Linear(self.out_features_dim[3]+self.out_features_dim[1]+self.out_features_dim[2] 
                            if not self.concat_word_emb else self.in_features_dim[-1], 200)

        self.final_GCN = GCN(200, self.out_features_dim[-1]).to(self.device)
        self.final_GCN_2 = GCN(self.out_features_dim[-1], self.out_features_dim[-1]).to(self.device)
        self.FC = nn.Linear(out_features_dim[-1], out_features_dim[0])

        self.adj = {}
        self.feature = {}

        for i in range(1, self.type_num):
            self.adj[str(0) + str(i)] = fetch_adj(self.adj_dict, 0, i, self.device)
            self.adj[str(i) + str(i)] = fetch_adj(self.adj_dict, i, i, self.device)
            self.feature[str(i)] = fetch_feature(self.features_dict, i, self.device)
    
    def forward(self, epoch):
        output = []
        for i in range(self.type_num - 1):
            if i == 1 and self.concat_word_emb:
                temp_emb = torch.cat([
                    F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)], identity=True)),
                                p=self.drop_out, training=self.training), self.features_dict['word_emb']], dim=-1)
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

        refined_text_input = [one for one in self.aggregate(output)]  # [type_num, query_num, emb_dim]

        norm = True
        if norm:
            refined_text_input_normed = []
            for i in range(self.type_num - 1):
                refined_text_input_normed.append(refined_text_input[i] / (refined_text_input[i].norm(p=2, dim=-1,keepdim=True) + 1e-9))
        else:
            refined_text_input_normed = refined_text_input

        cos_simi_total = torch.matmul(refined_text_input_normed[0], refined_text_input_normed[0].t()) + torch.matmul(
            refined_text_input_normed[1], refined_text_input_normed[1].t()) + torch.matmul(
            refined_text_input_normed[2], refined_text_input_normed[2].t())

        self.cos_simi_total = cos_simi_total

        refined_adj_tmp = cos_simi_total * (cos_simi_total > self.threshold).float()
        self.refined_adj_tmp = refined_adj_tmp

        refined_adj = refined_adj_tmp / (refined_adj_tmp.sum(dim=-1, keepdim=True) + 1e-9)
        refined_text_input_after_final_linear = self.refined_linear(torch.cat(refined_text_input_normed, dim=-1))
        refined_text_input_after_final_linear=F.dropout(refined_text_input_after_final_linear,p=self.drop_out, training=self.training)

        final_text_output_tmp=self.final_GCN(refined_adj,refined_text_input_after_final_linear)
        final_text_output = self.final_GCN_2(refined_adj, final_text_output_tmp)
        final_text_output = F.dropout(final_text_output, p=self.drop_out, training=self.training)

        self.final_text_output = final_text_output
        scores = self.FC(final_text_output)
        return scores

    def aggregate(self, input, softmax=False):
        aggregate_output = []
        for i in range(1, self.type_num):
            adj = self.adj[str(0) + str(i)]
            if softmax:
                adj = adj.masked_fill(adj.le(0), value=-1e9).softmax(-1)
            aggregate_output.append(torch.matmul(adj, input[i - 1]) / (torch.sum(adj, dim=-1, keepdim=True) + 1e-9))

        return aggregate_output


