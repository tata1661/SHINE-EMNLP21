# SHINE-EMNLP21

<p align="center"><img src="SHINE-thumbnail.png" alt="logo" width="600px" />


This repository provides the source codes of ["Hierarchical Heterogeneous Graph Representation Learning for Short Text Classification"](https://aclanthology.org/2021.emnlp-main.247/) published in *EMNLP 2021* as a long paper. 

Please cite our paper if you find it helpful. Thanks. 
```
@inproceedings{wang-etal-2021-hierarchical,
    title = "Hierarchical Heterogeneous Graph Representation Learning for Short Text Classification",
    author = "Wang, Yaqing  and
      Wang, Song  and
      Yao, Quanming  and
      Dou, Dejing",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.247",
    doi = "10.18653/v1/2021.emnlp-main.247",
    pages = "3091--3101",
    abstract = "Short text classification is a fundamental task in natural language processing. It is hard due to the lack of context information and labeled data in practice. In this paper, we propose a new method called SHINE, which is based on graph neural network (GNN), for short text classification. First, we model the short text dataset as a hierarchical heterogeneous graph consisting of word-level component graphs which introduce more semantic and syntactic information. Then, we dynamically learn a short document graph that facilitates effective label propagation among similar short texts. Thus, comparing with existing GNN-based methods, SHINE can better exploit interactions between nodes of the same types and capture similarities between short texts. Extensive experiments on various benchmark short text datasets show that SHINE consistently outperforms state-of-the-art methods, especially with fewer labels.",
}
```

## Environment  
We have implemented the method based on PyTorch and PaddlePaddle respectively, and both versions of the codes are open source.

Please prepare your environment according to the version you want to run.
### Torch Version:
- Python 3.7
- Pytorch 1.2

### Paddle Version:
- Python 3.7
- Paddlepaddle 2.2

## Quick Start
If you want to run the Torch Version:
```
cd SHINE-Torch
```
OR if you want to run the Paddle Version:
```
cd SHINE-Paddle
```

Then, You can quickly check out how SHINE operates on the Twitter dataset by using the command:
```
Python train.py
```

If you want to change the dataset, you can choose the specific dataset like: 
```
Python train.py --dataset snippets
```
Also, you can choose the specific GPU like:
```
Python train.py --dataset snippets --gpu 2
```

## Pretrained Embedding
    
Please visit https://drive.google.com/file/d/1gzIsN6XVqEXPJQR8MXVolbmKqlPgU_YA/view?usp=sharing to get the auxiliary pretrained embedding, including the NELL entity embedding and Glove6B word embedding. 

Then, you can check the preprocess folder for the code, and you can run the script:
```
Python preprocess.py
```
If you want to process your own data, you should adapt your data to the format in "snippets_split.json".


