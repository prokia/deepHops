# --*-- coding: utf-8 --*--

from functools import partial
from typing import List

from split_data import TASKS
import pickle
import numpy

protein_embedding = pickle.load(open('protein_emb.pkl', 'rb'))

def __get_emb(use_graph_embedding: bool, embedding_dim: int, label_list: List[int]) -> numpy.ndarray:
    if embedding_dim == 0:
        return numpy.array([0.0])
    if use_graph_embedding:
        return numpy.stack([protein_embedding[TASKS[index]] for index in label_list])
    else:
        # one_hot encoding
        v = [[0.0] * embedding_dim] * len(label_list)
        for i, label in enumerate(label_list):
            v[i][label] = 1.0
        return numpy.array(v)

__condition_transformer = None

def init_condition_transformer(use_graph_embedding, embedding_dim):
    global __condition_transformer
    __condition_transformer = partial(__get_emb, use_graph_embedding, embedding_dim)

def get_emb(label_list):
    return __condition_transformer(label_list)