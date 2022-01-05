import numpy as np
import pickle as pkl
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn import metrics
from model import SHINE

class Trainer(object):
    def __init__(self, params):
        self.dataset_name = params.dataset
        self.max_epoch = params.max_epoch
        self.save_path = params.save_path
        self.device = params.device
        self.hidden_size = params.hidden_size
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.concat_word_emb = params.concat_word_emb
        self.type_names = params.type_num_node
        self.file_dir = params.file_dir
        self.data_path = params.data_path

        self.adj_dict, self.features_dict, self.train_idx, self.test_idx, self.labels, self.nums_node = self.load_data()
        self.label_num = len(set(self.labels))
        self.labels = torch.tensor(self.labels).to(self.device)
        self.out_features_dim = [self.label_num, self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size]
        in_fea_final = self.out_features_dim[1] + self.out_features_dim[2] + self.out_features_dim[3]
        self.in_features_dim = [0, self.nums_node[1], self.nums_node[2], self.nums_node[-1], in_fea_final]

        if self.concat_word_emb: self.in_features_dim[-1] += self.features_dict['word_emb'].shape[-1]

        self.model = SHINE(self.adj_dict, self.features_dict, self.in_features_dim, self.out_features_dim, params)
        self.model = self.model.to(self.device)

        total_trainable_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_trainable_params:,} training parameters.')

        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optim = optim.Adam([{'params': self.model.parameters()},
                                 {'params': self.model.GCNs[0].parameters()},
                                 {'params': self.model.GCNs[2].parameters()},
                                 {'params': self.model.GCNs_2[2].parameters()},
                                 {'params': self.model.GCNs[1].parameters()},
                                 {'params': self.model.GCNs_2[0].parameters()},
                                 {'params': self.model.GCNs_2[1].parameters()}], lr=self.lr, weight_decay=self.weight_decay)

    def train(self):
        best_score = 0
        best_epoch = 0
        best_f1=0
        for i in range(1, self.max_epoch + 1):
            t=time.time()
            
            train_idx = self.train_idx
            output = self.model(i)
            train_scores = output[train_idx]
            train_labels = self.labels[train_idx]
            loss = F.cross_entropy(train_scores, train_labels)
            self.optim.zero_grad()
              
            loss.backward()
            self.optim.step()
            loss = loss.item()
            acc = torch.eq(torch.argmax(train_scores, dim=-1), train_labels).float().mean().item()

            if i%1==0:
                print('Epoch {}  loss: {:.4f} acc: {:.4f} time{:.4f}'.format(i, loss, acc,time.time()-t))
            if i % 5 == 0:
                acc_test, f1 = self.test()
                if acc_test > best_score:
                    best_score = acc_test
                    best_epoch = i
                    best_f1=f1
            if i%50==0:
                print('BEST SCORE', best_score, "BEST F1 SCORE",best_f1,'epoch', best_epoch)
        return best_score,best_f1

    def test(self):
        train_idx = self.train_idx

        self.model.training = False
        output = self.model(0)
        train_scores = output[train_idx]
        train_labels = self.labels[train_idx]

        with torch.no_grad():
            loss = F.cross_entropy(train_scores, train_labels)

            loss = loss.item()
            acc = torch.eq(torch.argmax(train_scores, dim=-1), train_labels).float().mean().item()

            test_scores = output[self.test_idx]
            test_labels = self.labels[self.test_idx]

            f1=metrics.f1_score(test_labels.detach().cpu().numpy(),torch.argmax(test_scores,-1).detach().cpu().numpy(),average='macro')

            loss_test = F.cross_entropy(test_scores, test_labels).item()
            acc_test = torch.eq(torch.argmax(test_scores, dim=-1), test_labels).float().mean().item()
            print('Test  loss: {:.4f} acc: {:.4f}'.format(loss, acc),
                  'loss: {:.4f} acc: {:.4f}'.format(loss_test, acc_test))
            

        self.model.training = True

        return acc_test, f1

    def load_data(self):
        adj_dict = {}
        feature_dict = {}
        nums_node = []
        for i in range(1, len(self.type_names)):
            adj_dict[str(0) + str(i)] = pkl.load(
                    open(self.data_path + './adj_{}2{}.pkl'.format(self.type_names[0], self.type_names[i]), 'rb'))
            if i == 1:
                nums_node.append(adj_dict[str(0) + str(i)].shape[0])
            if i != 3:
                adj_dict[str(i) + str(i)] = pkl.load(
                    open(self.data_path + './adj_{}.pkl'.format(self.type_names[i]), 'rb'))
                nums_node.append(adj_dict[str(i) + str(i)].shape[0])
            if i == 3:
                feature_dict[str(i)] = pkl.load(
                    open(self.data_path + './{}_emb.pkl'.format(self.type_names[i]), 'rb'))
                nums_node.append(feature_dict[str(i)].shape[0])
                nums_node.append(feature_dict[str(i)].shape[1])
            else:
                feature_dict[str(i)] = np.eye(nums_node[i], dtype=np.float64)

        feature_dict['word_emb'] = torch.tensor(pkl.load(
            open(self.data_path + './word_emb.pkl', 'rb')), dtype=torch.float).to(self.device)
        ent_emb=feature_dict['3']
        ent_emb_normed = ent_emb / np.sqrt(np.square(ent_emb).sum(-1, keepdims=True))
        adj_dict['33'] = np.matmul(ent_emb_normed, ent_emb_normed.transpose())
        adj_dict['33'] = adj_dict['33'] * np.float32(adj_dict['33'] > 0)
        
        adj_dict['22'] = np.array(adj_dict['22'].toarray())
        adj_dict['02'] = np.array(adj_dict['02'].toarray())
        adj_dict['03'] = np.array(adj_dict['03'].toarray())
        
        train_idx = json.load(open(self.data_path + './train_idx.json'.format(self.dataset_name)))
        test_idx = json.load(open(self.data_path + './test_idx.json'.format(self.dataset_name)))
        labels = json.load(open(self.data_path + './labels.json'.format(self.dataset_name)))

        select_40=False if self.dataset_name=='ohsu_title' else True
        kk=84
        if select_40:
            train_idx=[]
            valid_idx=[]
            test_idx=[]

            np.random.seed(kk)
            start_idx=np.random.random()*len(labels)*0.9
            for i in range(len(set(labels))):
                for j,label in enumerate (labels):
                    if j < start_idx:
                        continue

                    if label==i:
                        if len(train_idx)<(i+1)*20:
                            train_idx.append(j)
                        elif len(valid_idx)<(i+1)*20:
                            valid_idx.append(j)
            for j,label in enumerate (labels):
                if j not in train_idx and j not in valid_idx:
                    test_idx.append(j)
                    
            print(len(train_idx))
            print(len(valid_idx))
            print(len(test_idx))

        return adj_dict, feature_dict, train_idx, test_idx, labels, nums_node

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.model.state_dict(), path + './{}/save_model_new'.format(self.dataset_name))

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))
