import string
from abc import ABCMeta, abstractmethod
from typing import Dict, Set, Iterable, Tuple
from collections import defaultdict
from itertools import product

Word2Words = Dict[str, Set[str]]
Word2Typos = Word2Words
Typo2Words = Word2Words
Word2Backs = Word2Words


class Perturbator(metaclass=ABCMeta):
    @abstractmethod
    def perturb(self, words: Iterable[str]) -> Tuple[Word2Backs, Typo2Words]:
        pass


class Ed1Perturbator(Perturbator):
    """
    make perturbations within 1-edit distance
    all perturbations keep the first and last character unchanged
    """
    _default_perturb_type = 0b1111

    def __init__(self, **kwargs):
        self.perturb_type = kwargs['perturb_type'] if 'perturb_type' in kwargs else self._default_perturb_type
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        self._perturb_funcs = self._perturb_type(self.perturb_type)
        if self.verbose:
            print(self._perturb_funcs)

    def perturb(self, words: Iterable[str]) -> Tuple[Word2Backs, Typo2Words]:
        typo2words = defaultdict(set)
        word2backs = defaultdict(set)
        from tqdm import tqdm
        for word in tqdm(words, desc='perturb: ed1 perturb'):
            typos = self._perturb(word)
            typos.add(word)  # 将自身也加入到perturbation集合中
            for typo in typos:
                typo2words[typo].add(word)
        for typo, words in tqdm(typo2words, desc='perturb: conflict word'):
            if len(words) < 2:
                continue
            for w1, w2 in product(words, repeat=2):
                word2backs[w1].add(w2)
        return word2backs, typo2words

    def _perturb(self, word):
        word = str(word).lower()

        perturb_set = set()
        for func in self._perturb_funcs:
            perturb_words = func(word)
            perturb_set.update(perturb_words)
        return perturb_set

    @classmethod
    def _perturb_type(cls, type_=_default_perturb_type):
        _mask2perturb_func = {
            0b1000: cls._perturb_insert,
            0b0100: cls._perturb_delete,
            0b0010: cls._perturb_substitute,
            0b0001: cls._perturb_swap
        }
        funcs = []
        for mask, func in _mask2perturb_func.items():
            if mask & type_:
                funcs.append(func)
        return funcs

    @staticmethod
    def _perturb_insert(word):
        perturb_set = set()
        for i in range(1, len(word)):
            for c in string.ascii_lowercase:
                perturb_word = word[:i] + c + word[i:]
                perturb_set.add(perturb_word)
        return perturb_set

    @staticmethod
    def _perturb_delete(word):
        perturb_set = set()
        for i in range(1, len(word) - 1):
            perturb_word = word[:i] + word[i + 1:]
            perturb_set.add(perturb_word)
        return perturb_set

    @staticmethod
    def _perturb_substitute(word):
        perturb_set = set()
        for i in range(1, len(word) - 1):
            for c in string.ascii_lowercase:
                perturb_word = word[:i] + c + word[i + 1:]
                perturb_set.add(perturb_word)
        return perturb_set

    @staticmethod
    def _perturb_swap(word):
        perturb_set = set()
        for i in range(1, len(word) - 2):
            perturb_word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            perturb_set.add(perturb_word)
        return perturb_set


if __name__ == '__main__':
    toy = {'at', 'abet', 'aunt', 'about', 'abrupt'}
    ed1 = Ed1Perturbator(verbose=True)
    w2t, t2w = ed1.perturb(toy)
    t2w = {k: v for k, v in t2w.items() if len(v) >= 2}
    print(t2w)
