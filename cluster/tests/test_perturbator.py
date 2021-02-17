from ..perturbator import Ed1Perturbator
from itertools import combinations


def test_sys_path():
    import sys
    print(sys.path)
    assert 0


def test_ed1_only_one_word():
    words = ['word']
    ed1 = Ed1Perturbator(perturb_type=0b1111)
    w2t, t2w = ed1.perturb(words)
    assert set(w2t['word']) == set(t2w)
    for perturb in t2w:
        assert perturb.startswith('w') and perturb.endswith('d')
    insert_perturb = set([x for x in t2w if len(x) == 5])
    assert len(insert_perturb) == 3 * 26 - 2
    delete_perturb = set([x for x in t2w if len(x) == 3])
    assert len(delete_perturb) == 2
    others = set(t2w).difference(insert_perturb).difference(delete_perturb)
    for item in others:
        assert len(item) == 4


def test_ed1_one_connect_component():
    words = ['at', 'abet', 'aunt', 'about', 'abrupt', 'eat', 'out']
    ed1 = Ed1Perturbator(perturb_type=0b1111)
    w2t, t2w = ed1.perturb(words)
    conflicts = set([tuple(sorted(v)) for k, v in t2w.items() if len(v) >= 2])
    print(conflicts)
    #
    for pair in combinations(['at', 'eat', 'out'], r=2):
        pair = tuple(sorted(pair))
        assert pair not in conflicts
    #
    for pair in combinations(['at', 'abet', 'aunt'], r=2):
        pair = tuple(sorted(pair))
        assert pair in conflicts
