from .vocabulary import Vocabulary
from .perturbator import Perturbator

from typing import Dict, Set
from itertools import combinations
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from collections import defaultdict


class OriginCluster:
    def __init__(self, vocab: Vocabulary, perturbator: Perturbator):
        self.vocab = vocab
        self.perturbator = perturbator
        #
        self.word2backs, self.typo2words = perturbator.perturb(vocab.idx2vocab)
        #
        self._graph = self._make_graph()
        #
        self.clusters: Dict[int, Set[str]] = defaultdict(set)
        self.word2cluster: Dict[str, int] = dict()
        self.cluster2representative: Dict[int, str] = dict()

    def _make_graph(self):
        shape = (len(self.vocab), len(self.vocab))
        graph = sparse.dok_matrix(shape, dtype=bool)
        from tqdm import tqdm
        for word, backs in tqdm(self.word2backs.items(), desc='making conflict graph'):
            for back in backs:
                row = self.vocab.vocab2idx[word]
                col = self.vocab.vocab2idx[back]
                graph[row, col] = True
        graph = (graph + graph.T).tocsr()
        return graph

    def make_cluster(self):
        _, components_list = connected_components(self._graph)
        from tqdm import tqdm
        for idx, cluster in enumerate(tqdm(components_list, desc='making cluster')):
            word = self.vocab.idx2vocab[idx]
            if cluster not in self.cluster2representative:
                self.cluster2representative[cluster] = word
            self.clusters[cluster].add(word)
            self.word2cluster[word] = cluster

    def show_cluster_summary(self):
        from collections import Counter
        if len(self.clusters) == 0:
            print('no clusters')
            return
        print(f'num of clusters: {len(self.clusters)}')
        print(f'biggest  cluster\'s size is: {max(len(items) for items in self.clusters.values())}')
        print(f'mode     cluster\'s size is: {Counter(len(items) for items in self.clusters.values()).most_common(1)}')
        print(f'smallest cluster\'s size is: {min(len(items) for items in self.clusters.values())}')