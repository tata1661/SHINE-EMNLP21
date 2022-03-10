import random
import numpy as np
import paddle
import os
import json

def fetch_to_tensor(dicts, dict_type, device):
    return paddle.to_tensor(dicts[dict_type], dtype='float64', place=device)

def aggregate(adj_dict, input, other_type_num, softmax=False):
    aggregate_output = []
    for i in range(other_type_num):
        adj = adj_dict[str(0) + str(i + 1)]
        if softmax:
            adj = adj.masked_fill(adj.le(0), value=-1e9).softmax(-1)
        tmp = paddle.matmul(adj, input[i]) / (paddle.sum(adj, axis=1, keepdim=True) + 1e-9)
        aggregate_output.append(tmp)
    return aggregate_output

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    paddle.seed(seed)

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
