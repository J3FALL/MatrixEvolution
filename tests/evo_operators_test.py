import numpy as np

from evo_operators import (
    fitness_frob_norm,
    random_matrix,
    single_point_crossover
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
    matrix = random_matrix(matrix_size=expected_size)
    actual_size = matrix.shape

    assert expected_size == actual_size


def test_single_point_crossover_correct():
    parent_first, parent_second = random_matrix(matrix_size=(3, 3)), random_matrix(matrix_size=(3, 3))

    child_first, child_second = single_point_crossover(parent_first, parent_second, 'vertical')

    print(f'Parent first: {parent_first}')
    print(f'Parent second: {parent_second}')
    print(f'Child first: {child_first}')
    print(f'Child second{child_second}')

    assert parent_first.shape == child_first.shape == child_second.shape
