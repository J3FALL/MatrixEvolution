import numpy as np

from evo_algorithm import (
    select_k_best,
    initial_population_from_lhs_only_s
)


def test_select_k_bst_correct():
    candidates = np.random.randint(100, size=(100,))
    graded = sorted(candidates)
    expected_best = graded[0]
    top_k_best = select_k_best(candidates=graded, k=25)

    actual_best = top_k_best[0]

    assert expected_best == actual_best
    assert len(top_k_best) == 25


def test_initial_population_from_lhs_only_s_correct():
    matrix = np.random.rand(10, 10)
    samples_total = 50
    samples = initial_population_from_lhs_only_s(samples_amount=samples_total, vector_size=matrix.shape[0],
                                                 values_range=10, source_matrix=matrix)

    assert len(samples) == samples_total
