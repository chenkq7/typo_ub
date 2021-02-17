from typing import Dict, Set, List


class Vocabulary:
    def __init__(self, vocab: Set[str], word2freq: Dict[str, int], top_freq=100_000):
        word2freq_filtered = {k: v for k, v in word2freq.items() if k in vocab}
        word2freq_filtered = sorted(word2freq_filtered.items(), key=lambda x: x[1], reverse=True)[:top_freq]
        self.size = len(word2freq_filtered)
        self.word2freq: Dict[str, int] = dict(word2freq_filtered)
        self.idx2vocab: List[str] = [word for word, freq in word2freq_filtered]
        self.vocab2idx: Dict[str, int] = {word: i for i, word in enumerate(self.idx2vocab)}

    def __len__(self):
        return self.size


if __name__ == '__main__':
    from io_wrapper import get_glove_vocab, get_coca_word_frequency

    toy_vocab = '../data/toy_vocab.txt'
    toy_w2f = '../data/toy_word2freq.json'
    toy_vocab = get_glove_vocab(toy_vocab)
    toy_w2f = get_coca_word_frequency(toy_w2f)

    vocab = Vocabulary(toy_vocab, toy_w2f)