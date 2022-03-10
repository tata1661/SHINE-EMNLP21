import os 
import torch
import random
import numpy as np
import json


def fetch_to_tensor(dicts, dict_type, device):
    return torch.tensor(dicts[dict_type], dtype=torch.float, device=device)

def aggregate(adj_dict, input, other_type_num, softmax=False):
    aggregate_output = []
    for i in range(other_type_num):
        adj = adj_dict[str(0) + str(i + 1)]

        if softmax:
            adj = adj.masked_fill(adj.le(0), value=-1e9).softmax(-1)
        aggregate_output.append(torch.matmul(adj, input[i]) / (torch.sum(adj, dim=-1).unsqueeze(-1) + 1e-9))
    return aggregate_output


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    os.makedirs(params.save_path, exist_ok=True)
    fname = params.save_name
    if not os.path.isfile(fname):
        with open(fname, mode='w') as f:
            f.write(json.dumps({str(k): result[k] for k in result}, cls=MyEncoder, indent=4))
            f.close()
    else:
        with open(fname, mode='a') as f:
            f.write(json.dumps({str(k): result[k] for k in result}, cls=MyEncoder, indent=4))
            f.close()