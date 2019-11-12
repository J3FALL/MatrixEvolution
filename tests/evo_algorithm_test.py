import numpy as np

from evo_algorithm import select_k_best


def test_select_k_bst_correct():
    candidates = np.random.randint(100, size=(100,))
    graded = sorted(candidates)
    expected_best = graded[0]
    top_k_best = select_k_best(candidates=graded, k=25)

    actual_best = top_k_best[0]

    assert expected_best == actual_best
    assert len(top_k_best) == 25
