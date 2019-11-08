import numpy as np

from evo_operators import (
    fitness_frob_norm,
    new_individ_random
)


def test_frob_norm_correct():
    source = np.random.rand(100, 100)
    u, s, vh = np.linalg.svd(source, full_matrices=True)
    actual = np.dot(u * s, vh)

    frob_norm = np.linalg.norm(source - actual)

    assert np.allclose(frob_norm, 0.0, 10e-8)


def test_fitness_frob_norm_correct():
    source = np.random.rand(100, 100)
    u, s, vh = np.linalg.svd(source, full_matrices=True)

    value = fitness_frob_norm(source_matrix=source, svd=(u, s, vh))

    assert np.allclose(value, 0.0, 10e-8)


def test_new_individ_random_correct():
    expected_size = (10, 10)
    matrix = new_individ_random(matrix_size=expected_size)
    actual_size = matrix.shape

    assert expected_size == actual_size
