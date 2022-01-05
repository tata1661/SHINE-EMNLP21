import os 
import torch
import random
import numpy as np
import json

def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0),device=A.device)
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D).to(device=A.device)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A).to(device=A.device)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fetch_adj(adj_dict, row_type, col_type, device):
    result = torch.tensor(adj_dict[str(row_type) + str(col_type)], dtype=torch.float, device=device)
    return result

def fetch_feature(features_dict, type, device):
    result = torch.tensor(features_dict[str(type)], dtype=torch.float, device=device)
    return result

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self,obj)

def save_res(params, acc, f1):
    from collections import defaultdict
    result=defaultdict(list)
    result[tuple([acc,f1])] = {
                                        'seed': params.seed,
                                        'weigh_dacay': params.weight_decay, 
                                        'lr': params.lr,
                                        'drop_out': params.drop_out,
                                        'threshold': params.threshold,
                                        }

    fname = params.save_name
    if not os.path.isfile(fname):
        with open(fname, mode='w') as f:
            f.write(json.dumps({str(k): result[k] for k in result}, cls=MyEncoder, indent=4))
            f.close()
    else:
        with open(fname, mode='a') as f:
            f.write(json.dumps({str(k): result[k] for k in result}, cls=MyEncoder, indent=4))
            f.close()