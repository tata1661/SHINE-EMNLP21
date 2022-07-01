import json
import numpy as np
import pickle as pkl
import math
import nltk
from tqdm import tqdm
import os

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import re
def clean_str(string,use=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if not use: return string

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_stopwords(filepath='./stopwords_en.txt'):
    stopwords = set()
    with open(filepath, 'r') as f:
        for line in f:
            swd = line.strip()
            stopwords.add(swd)
    print(len(stopwords))
    return stopwords

def tf_idf_transform(inputs, mapping=None, sparse=False):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from scipy.sparse import coo_matrix
    vectorizer = CountVectorizer(vocabulary=mapping)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(inputs))
    weight = tf_idf.toarray()
    return weight if not sparse else coo_matrix(weight)

def PMI(inputs, mapping, window_size, sparse):
    W_ij = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
    W_i = np.zeros([len(mapping)], dtype=np.float64)
    W_count = 0
    for one in inputs:
        word_list = one.split(' ')
        if len(word_list) - window_size < 0:
            window_num = 1
        else:
            window_num = len(word_list) - window_size + 1
        for i in range(window_num):
            W_count += 1
            context = list(set(word_list[i:i + window_size]))
            while '' in context:
                context.remove('')
            for j in range(len(context)):
                W_i[mapping[context[j]]] += 1
                for k in range(j + 1, len(context)):
                    W_ij[mapping[context[j]], mapping[context[k]]] += 1
                    W_ij[mapping[context[k]], mapping[context[j]]] += 1
    if sparse:
        rows = []
        columns = []
        data = []
        for i in range(len(mapping)):
            rows.append(i)
            columns.append(i)
            data.append(1)
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i]
            for j in tmp:
                value = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if value > 0:
                    rows.append(i)
                    columns.append(j)
                    data.append(value)
                    rows.append(j)
                    columns.append(i)
                    data.append(value)
        PMI_adj = coo_matrix((data, (rows, columns)), shape=(len(mapping), len(mapping)))
    else:
        PMI_adj = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
        for i in range(len(mapping)):
            PMI_adj[i, i] = 1  
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i] 
            # for j in range(i + 1, len(mapping)):
            for j in tmp:
                pmi = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if pmi > 0:
                    PMI_adj[i, j] = pmi
                    PMI_adj[j, i] = pmi
    return PMI_adj

def make_node2id_eng_text(dataset_name, remove_StopWord=False):
    stop_word=load_stopwords()
    stop_word.add('')
    os.makedirs(f'./{dataset_name}_data', exist_ok=True)

    f_train = json.load(open('./{}_split.json'.format(dataset_name)))['train']
    f_test = json.load(open('./{}_split.json'.format(dataset_name)))['test']

    from collections import defaultdict
    word_freq=defaultdict(int)
    for item in f_train.values():
        words=clean_str(item['text']).split(' ')
        for one in words:
            word_freq[one.lower()]+=1
    for item in f_test.values():
        words=clean_str(item['text']).split(' ')
        for one in words:
            word_freq[one.lower()]+=1
    freq_stop=0
    for word,count in word_freq.items():
        if count<5:
            stop_word.add(word)
            freq_stop+=1
    print('freq_stop num',freq_stop)

    ent2id_new = json.load(open('./pretrained_emb/NELL_KG/ent2ids_refined', 'r'))
    adj_ent_index = []
    query_nodes = []
    tag_set = set()
    entity_set = set()
    words_set = set()
    train_idx = []
    test_idx = []
    labels = []
    tag_list = []
    word_list = []
    ent_mapping = {} 

    for i, item in enumerate(tqdm(f_train.values())):
        # item=f_train[str(i)]
        query = clean_str(item['text'])
        if not query:
            print(query)
            continue
        tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]
        if '' in tags:
            print(item)

        tag_list.append(' '.join(tags))
        tag_set.update(tags)
        labels.append(item['label'])
        if remove_StopWord:
            words = [one.lower() for one in query.split(' ') if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(' ')]
        if '' in words:
            print(words)

        ent_list = []
        index = [] 
        for key in ent2id_new.keys():
            if key in query.lower():
                ent_list.append(key)
                if key not in ent_mapping:
                    ent_mapping[key] = len(ent_mapping)
                    entity_set.update(ent_list)
                if ent_mapping[key] not in index: index.append(ent_mapping[key])
        # print(entity_set)
        adj_ent_index.append(index)
        word_list.append(' '.join(words))
        words_set.update(words)
        if query:
            query_nodes.append(query)
        else:
            print(item)
            print(query)
        train_idx.append(len(train_idx))

    for i, item in enumerate(tqdm(f_test.values())):
        # item = f_test[str(i)]
        query = clean_str(item['text'])
        # print(query)
        if not query:
            print(query)
            continue
        tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]
        tag_list.append(' '.join(tags))
        tag_set.update(tags)
        labels.append(item['label'])
        if remove_StopWord:
            words = [one.lower() for one in query.split(' ') if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(' ')]
        if '' in words:
            print(words)
        ent_list = []
        index = []
        for key in ent2id_new.keys():
            if key in query.lower():
                if key not in ent_mapping:
                    ent_list.append(key)
                    ent_mapping[key] = len(ent_mapping)
                    entity_set.update(ent_list)
                if ent_mapping[key] not in index: index.append(ent_mapping[key])
        adj_ent_index.append(index)

        word_list.append(' '.join(words))
        words_set.update(words)
        if query:
            query_nodes.append(query)
        else:
            print(item)
            print(query)
            print(query)

        test_idx.append(len(test_idx)+ len(train_idx))

    print(tag_set)
    json.dump([adj_ent_index, ent_mapping],
              open('./{}_data/index_and_mapping.json'.format(dataset_name), 'w'), ensure_ascii=False)
    ent_emb = []
    TransE_emb_file = np.loadtxt('./pretrained_emb/NELL_KG/entity2vec.TransE')
    TransE_emb = []

    for i in range(len(TransE_emb_file)):
        TransE_emb.append(list(TransE_emb_file[i, :]))

    rows = []
    data = []
    columns = []

    max_num = len(ent_mapping)
    for i, index in enumerate(adj_ent_index):
        for ind in index:
            data.append(1)
            rows.append(i)
            columns.append(ind)

    adj_ent = coo_matrix((data, (rows, columns)), shape=(len(adj_ent_index), max_num))
    for key in ent_mapping.keys():
        ent_emb.append(TransE_emb[ent2id_new[key]])

    ent_emb = np.array(ent_emb)
    print('ent shape', ent_emb.shape)
    ent_emb_normed = ent_emb / np.sqrt(np.square(ent_emb).sum(-1, keepdims=True))
    adj_emb = np.matmul(ent_emb_normed, ent_emb_normed.transpose())
    print('entity_emb_cos', np.mean(np.mean(adj_emb, -1)))
    pkl.dump(np.array(ent_emb), open('./{}_data/entity_emb.pkl'.format(dataset_name), 'wb'))
    pkl.dump(adj_ent, open('./{}_data/adj_query2entity.pkl'.format(dataset_name), 'wb'))

    word_nodes = list(words_set)
    tag_nodes = list(tag_set)
    entity_nodes = list(entity_set)
    # nodes_all = list(query_nodes | tag_nodes | entity_nodes)
    nodes_all = query_nodes + tag_nodes + entity_nodes + word_nodes
    nodes_num = len(query_nodes) + len(tag_nodes) + len(entity_nodes) + len(word_nodes)
    print('query', len(query_nodes))
    print('tag', len(tag_nodes))
    print('ent', len(entity_nodes))
    print('word', len(word_nodes))

    if len(nodes_all) != nodes_num:
        print('duplicate name error')

    print('len_train',len(train_idx))
    print('len_test',len(test_idx))
    print('len_quries',len(query_nodes))

    tags_mapping = {key: value for value, key in enumerate(tag_nodes)}
    words_mapping = {key: value for value, key in enumerate(word_nodes)}
    adj_query2tag = tf_idf_transform(tag_list, tags_mapping)
    adj_tag = PMI(tag_list, tags_mapping, window_size=5, sparse=False)
    pkl.dump(adj_query2tag, open('./{}_data/adj_query2tag.pkl'.format(dataset_name), 'wb'))
    pkl.dump(adj_tag, open('./{}_data/adj_tag.pkl'.format(dataset_name), 'wb'))
    adj_query2word = tf_idf_transform(word_list, words_mapping, sparse=True)
    adj_word = PMI(word_list, words_mapping, window_size=5, sparse=True)
    pkl.dump(adj_query2word, open('./{}_data/adj_query2word.pkl'.format(dataset_name), 'wb'))
    pkl.dump(adj_word, open('./{}_data/adj_word.pkl'.format(dataset_name), 'wb'))
    json.dump(train_idx, open('./{}_data/train_idx.json'.format(dataset_name), 'w'), ensure_ascii=False)
    json.dump(test_idx, open('./{}_data/test_idx.json'.format(dataset_name), 'w'), ensure_ascii=False)

    label_map = {value: i for i, value in enumerate(set(labels))}
    json.dump([label_map[label] for label in labels], open('./{}_data/labels.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(query_nodes, open('./{}_data/query_id2_list.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(tag_nodes, open('./{}_data/tag_id2_list.json'.format(dataset_name), 'w'), ensure_ascii=False)
    json.dump(entity_nodes, open('./{}_data/entity_id2_list.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(word_nodes, open('./{}_data/word_id2_list.json'.format(dataset_name), 'w'), ensure_ascii=False)

    glove_emb = pkl.load(open('./pretrained_emb/old_glove_6B/embedding_glove.p', 'rb'))
    vocab = pkl.load(open('./pretrained_emb/old_glove_6B/vocab.pkl', 'rb'))
    embs = []
    err_count = 0
    for word in word_nodes:
        if word in vocab:
            embs.append(glove_emb[vocab[word]])
        else:
            err_count += 1
            # print('error:', word)
            embs.append(np.zeros(300, dtype=np.float64))
    print('err in word count', err_count)
    pkl.dump(np.array(embs, dtype=np.float64), open('./{}_data/word_emb.pkl'.format(dataset_name), 'wb'))

dataset_name='snippets'
if dataset_name in ['mr', 'snippets', 'tagmynews']:
    remove_StopWord = True
else:
    remove_StopWord = False
make_node2id_eng_text(dataset_name, remove_StopWord)