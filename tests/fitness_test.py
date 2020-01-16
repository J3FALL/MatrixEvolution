import numpy as np

from evo.operators.fitness import (
    fitness_svd_frob_norm,
    fitness_inverse_matrix_frob_norm
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

    value = fitness_svd_frob_norm(source_matrix=source, genotype=(u, s, vh))

    assert np.allclose(value, 0.0, 10e-8)


def test_fitness_inverse_matrix_frob_norm_correct():
    matrix_size = 100
    source = np.random.rand(matrix_size, matrix_size)
    invert_matrix = np.linalg.inv(source)

    value = fitness_inverse_matrix_frob_norm(source_matrix=source,
                                             genotype=invert_matrix)

    assert  np.allclose(value, 0.0, 10e-8)
