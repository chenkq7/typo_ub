from typing import Dict, List, Set, Any
from collections import defaultdict
import numpy as np


class Agglomerate:
    def __init__(self, gamma, words: List[str], word2freq: Dict[str, int], word2embedding: Dict[str, Any],
                 back_words: Dict[str, Set[str]]):
        # non changeable
        self.gamma = gamma
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2freq = np.asarray([word2freq[word] for word in words])
        self.idx2embedding = np.asarray([word2embedding[word] for word in words])
        # 将back_words映射成back_idxes: 只映射在words中的单词
        self.idx2back_idxes: List[Set[int]] = [set(map(lambda w: self.word2idx.get(w, self.word2idx[word]), back_words[word]))
                                               for word in words]
        self.word_num, self.embedding_dim = self.idx2embedding.shape
        # init
        self.idx2cluster = np.arange(len(self.word2idx))
        # init cache
        self._update_cache()

    def solve(self):
        while True:
            delta, cluster_min, cluster_max = self.find_better_result_by_delta()
            assert cluster_min is None or cluster_min < cluster_max
            if delta <= 0:
                break
            self.merge_cluster(cluster_min, cluster_max)
            self._update_cache()

    def merge_cluster(self, cluster_min, cluster_max):
        for idx in range(self.word_num):
            if self.idx2cluster[idx] == cluster_max:
                self.idx2cluster[idx] = cluster_min

    def _update_cache(self):
        self.idx2back_cluster = np.asarray([set(map(self.idx2cluster.__getitem__, self.idx2back_idxes[idx]))
                                            for idx in range(self.word_num)])
        self.cluster2total_freq = self._accumulate_by_cluster(self.idx2freq, origin_value=0)
        idx2cluster_total_freq = np.asarray(self._broadcast_by_cluster(self.cluster2total_freq))
        weight = self.idx2freq / idx2cluster_total_freq
        weighted_embedding = weight.reshape(-1, 1) * self.idx2embedding
        self.cluster2mu = self._accumulate_by_cluster(weighted_embedding, origin_value=np.zeros(self.embedding_dim))
        idx2mu = np.asarray(self._broadcast_by_cluster(self.cluster2mu))
        self.idx2distance = np.linalg.norm(self.idx2embedding - idx2mu, axis=-1)

    def _accumulate_by_cluster(self, v, origin_value: Any = 0, acc_func=lambda acc, x: acc + x) -> Dict[int, Any]:
        ans = defaultdict(lambda: origin_value)
        for idx, cluster in enumerate(self.idx2cluster):
            ans[cluster] = acc_func(ans[cluster], v[idx])
        return ans

    def _broadcast_by_cluster(self, cluster2value: Dict[int, Any]):
        ans = [0] * len(self.idx2cluster)
        for idx, cluster in enumerate(self.idx2cluster):
            ans[idx] = cluster2value[cluster]
        return ans

    def find_better_result_by_delta(self):
        max_delta, delta_cluster_min, delta_cluster_max = 0, None, None
        queried = set()
        for idx1, idxes in enumerate(self.idx2back_idxes):
            for idx2 in idxes:
                cluster1, cluster2 = sorted([self.idx2cluster[idx1], self.idx2cluster[idx2]])
                # assert cluster1 < cluster2
                if cluster1 == cluster2 or (cluster1, cluster2) in queried:
                    continue
                queried.add((cluster1, cluster2))
                # idx2back_cluster is one-hot idx vector for cluster
                idx2back_cluster1 = set(idx for idx in range(self.word_num) if cluster1 in self.idx2back_cluster[idx])
                idx2back_cluster2 = set(idx for idx in range(self.word_num) if cluster2 in self.idx2back_cluster[idx])
                idx2back_cluster1 = np.asarray([(idx in idx2back_cluster1) for idx in range(self.word_num)])
                idx2back_cluster2 = np.asarray([(idx in idx2back_cluster2) for idx in range(self.word_num)])
                # calculate delta stab if merge cluster1 and cluster2
                intersect_idx = np.logical_and(idx2back_cluster1, idx2back_cluster2)
                delta_stability = intersect_idx @ self.idx2freq
                assert delta_stability >= 0
                # calculate delta fid if merge cluster1 and cluster2
                mu_cluster1, mu_cluster2 = self.cluster2mu[cluster1], self.cluster2mu[cluster2]
                freq_cluster1, freq_cluster2 = self.cluster2total_freq[cluster1], self.cluster2total_freq[cluster2]
                mu_merged = (freq_cluster1 * mu_cluster1 + freq_cluster2 * mu_cluster2) / (freq_cluster1 + freq_cluster2)
                distance_merged = np.linalg.norm(self.idx2embedding - mu_merged, axis=-1)  # only has meaning for idx in union_idx
                union_idx = np.logical_or(idx2back_cluster1, idx2back_cluster2)
                fid_merged = (union_idx * distance_merged) @ self.idx2freq
                fid_cluster1 = (idx2back_cluster1 * self.idx2distance) @ self.idx2freq
                fid_cluster2 = (idx2back_cluster2 * self.idx2distance) @ self.idx2freq
                delta_fidelity = fid_cluster1 + fid_cluster2 - fid_merged
                assert delta_fidelity <= 0
                # judge whether merge brings profit
                new_delta = self.gamma * delta_fidelity + (1 - self.gamma) * delta_stability
                if new_delta > max_delta:
                    max_delta, delta_cluster_min, delta_cluster_max = new_delta, cluster1, cluster2
        return max_delta, delta_cluster_min, delta_cluster_max

    def object_func(self):
        # stability
        idx2back_cluster_num = np.asarray([len(self.idx2back_cluster[idx]) for idx in range(self.word_num)])
        stability = - idx2back_cluster_num @ self.idx2freq
        # fidelity
        fidelity = - self.idx2distance @ self.idx2freq
        # tradeoff by gamma
        return self.gamma * fidelity + (1 - self.gamma) * stability
