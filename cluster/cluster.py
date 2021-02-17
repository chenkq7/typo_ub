from typing import Dict, Set, Tuple
from abc import ABCMeta, abstractmethod
from collections import defaultdict

Word2Freq = Dict[str, int]
Word2Rep = Dict[str, str]
Rep2Cluster = Dict[str, Set[str]]


class Cluster:
    def __init__(self):
        self.word2freq: Word2Freq = dict()
        self.word2represent: Word2Rep = dict()
        self.rep2cluster: Rep2Cluster = defaultdict(set)

    def add_cluster(self, word2rep: Word2Rep):
        duplicated = set(word2rep).intersection(set(self.word2represent))
        assert not duplicated, f'some words exist:{duplicated}'
        self.word2represent.update(word2rep)
        for word, rep in word2rep.items():
            self.rep2cluster[rep].add(word)

    def assign_freq(self, word2freq: Word2Freq, default_freq=None):
        self.word2freq.clear()
        if default_freq is None:
            for word in self.word2represent:
                self.word2freq[word] = word2freq[word]
        else:
            for word in self.word2represent:
                self.word2freq[word] = word2freq[word] if word in word2freq else default_freq

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.state_dict(), f)

    def state_dict(self):
        return self.__dict__


class ClusterMaker(metaclass=ABCMeta):
    @abstractmethod
    def make_cluster(self) -> Tuple[Word2Rep, Rep2Cluster]:
        pass


if __name__ == '__main__':
    cluster = Cluster()
    cluster.add_cluster({'apple': 'apple', 'at': 'at', 'abet': 'at'})
    print(cluster.state_dict())
