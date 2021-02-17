from cluster.io_wrapper import get_glove_vocab, get_coca_word_frequency
from cluster.vocabulary import Vocabulary
from cluster.perturbator import Ed1Perturbator
from cluster.origincluster import OriginCluster


def main(**kwargs):
    glove_file = str(kwargs['glove_file'])
    coca_file = str(kwargs['coca_file'])
    output_file = str(kwargs['output_prefix'])
    perturb_args = kwargs['perturb_args']
    top_freq = kwargs['top_freq']

    glove_vocab = get_glove_vocab(glove_file)
    coca_w2f = get_coca_word_frequency(coca_file)
    print(f'len(glove_vocab): {len(glove_vocab)}')
    print(f'len(coca_word2freq): {len(coca_w2f)}')

    vocab = Vocabulary(glove_vocab, coca_w2f, top_freq=top_freq)
    print(f'final vocab size: {len(vocab)}')
    ed1 = Ed1Perturbator(**perturb_args)
    cluster = OriginCluster(vocab, ed1)
    cluster.make_cluster()
    cluster.show_cluster_summary()

    output_file += f'_cluster_{len(vocab)}_{ed1.perturb_type}.pkl'
    with open(output_file, 'wb') as fp:
        import pickle
        pickle.dump(cluster, fp)


if __name__ == '__main__':
    from pathlib import Path

    dir_path = Path(__file__).parent

    toy_args = dict(
        glove_file=dir_path / './data/toy_vocab.txt',
        coca_file=dir_path / './data/toy_word2freq.json',
        output_prefix=dir_path / './data/toy',
        perturb_args={},
        top_freq=100_000,
    )

    normal_args = dict(
        glove_file=dir_path / './data/glove.6B.50d.txt',
        coca_file=dir_path / './data/coca-1grams.json',
        output_prefix=dir_path / './data/glove_coca',
        perturb_args={},
        top_freq=100_000,
    )

    print(toy_args)
    main(**toy_args)
