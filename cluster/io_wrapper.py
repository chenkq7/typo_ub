import json
from typing import Dict, Set
from collections import defaultdict


def get_glove_vocab(glove_file: str) -> Set[str]:
    with open(glove_file, 'r', encoding='utf8')as f:
        words = [line.strip().split()[0] for line in f.readlines()]
    vocab = set(words)
    return vocab


def get_coca_word_frequency(coca_file: str) -> Dict[str, int]:
    word2freq = defaultdict(lambda: 0)
    with open(coca_file, 'r', encoding='ISO-8859-1') as f:
        coca_freq_dict = json.load(f)
    for elem in coca_freq_dict:
        word, num = elem.split('_')[0], coca_freq_dict[elem]
        word2freq[word] += int(num)
    return word2freq


if __name__ == '__main__':
    # GLOVE_FILE = '../data/glove.6B.50d.txt'
    # COCA_FILE = '../data/coca-1grams.json'
    # vocab_ = get_glove_vocab(GLOVE_FILE)
    # word2freq_ = get_coca_word_frequency(COCA_FILE)

    toy_vocab = '../data/toy_vocab.txt'
    toy_w2f = '../data/toy_word2freq.json'
    vocab_ = get_glove_vocab(toy_vocab)
    word2freq_ = get_coca_word_frequency(toy_w2f)

