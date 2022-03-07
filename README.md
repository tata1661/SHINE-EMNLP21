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

- Python 3.7
- Pytorch 1.2
- Paddlepaddle 2.2


## Quick Start
Torch Version:
```
cd SHINE-Torch
```
or 
Paddle Version:
```
cd SHINE-Paddle
```
You can quickly check out how SHINE operates on the Twitter dataset by using the command:
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

## Data Pre-processing
    
Please check the data process folder for more information, you can run the script:
```
Python DataProcess.py
```
If you want to process your own data, you should adapt your data to the format in "snippets_split.json".


