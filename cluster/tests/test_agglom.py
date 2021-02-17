from ..agglom_cluster import Agglomerate
from ..perturbator import Ed1Perturbator
from collections import defaultdict
from itertools import permutations
import numpy as np


def test_agglomerate():
    cluster_dict = {'cluster': {0: ['stop', 'step'], 1: ['plain', 'pin', 'pun'], 2: ['ham']},
                    'word2cluster': {'stop': 0, 'step': 0, 'plain': 1, 'pin': 1, 'pun': 1, 'ham': 2},
                    'word2freq': {'stop': 100, 'step': 50, 'plain': 75, 'pin': 15, 'pun': 10, 'ham': 5},
                    'cluster2representative': {0: 'stop', 1: 'plain', 2: 'ham'}}

    conflict_dict = {'stop': {'step', 'stop'},
                     'step': {'stop', 'step'},
                     'pin': {'pun', 'pin'},
                     'pun': {'pin', 'pun'},
                     'plain': {'plain'},
                     'ham': {'ham'}
                     }

    gamma_0_cluster_max = {
        0: 0,
        1: 1,
        2: 0
    }

    gamma = 0
    word2freq = cluster_dict['word2freq']
    for key, words in cluster_dict['cluster'].items():
        embedding = np.eye(len(words))
        word2embedding = {word: embedding[idx] for idx, word in enumerate(words)}
        agg = Agglomerate(gamma, words, word2freq, word2embedding, conflict_dict)
        agg.solve()
        print(agg.word2idx)
        print(agg.idx2cluster)
        assert (max(agg.idx2cluster) == gamma_0_cluster_max[key])


def test_agglomerate_2():
    words = ['apple', 'banana', 'at', 'aunt', 'abet', 'about', 'abrupt']
    freqs = [1, 2, 100, 2, 70, 10, 40]
    word2freq = dict(zip(words, freqs))
    embedding = np.eye(len(words))
    word2embedding = {word: embedding[idx] for idx, word in enumerate(words)}
    back_words = {
        'apple': {'apple'},
        'banana': {'banana'},
        'at': {'at', 'abet', 'aunt'},
        'abet': {'at', 'abet', 'aunt', 'about'},
        'aunt': {'at', 'abet', 'aunt'},
        'abrupt': {'abrupt', 'about'},
        'about': {'abrupt', 'about', 'abet'},
    }
    for gamma in np.linspace(0, 1, 30):
        agg = Agglomerate(gamma, words, word2freq, word2embedding, back_words)
        agg.solve()
        print(f"{gamma} : {agg.idx2cluster}")
