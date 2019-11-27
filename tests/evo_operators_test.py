import numpy as np

from evo_operators import (
    fitness_frob_norm,
    random_matrix,
    single_point_crossover,
    mutation_gauss,
    two_point_crossover,
    k_point_crossover,
    initial_pop_flat_lhs_only_u,
    initial_population_only_u_rotations
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


def test_mutation_gauss_correct():
    candidate = random_matrix(matrix_size=(10, 10))
    source_shape = candidate.shape
    mu, sigma = 0, 0.1
    prob_global = 0.05
    resulted = mutation_gauss(candidate=candidate, mu=mu, sigma=sigma, prob_global=prob_global)

    assert source_shape == resulted.shape


def test_two_point_crossover_correct():
    parent_first, parent_second = np.zeros((5, 5)), np.ones((5, 5))

    child_first, child_second = two_point_crossover(parent_first, parent_second, 'horizontal')

    print(f'Child first: {child_first}')
    print(f'Child second: {child_second}')

    assert parent_first.shape == child_first.shape == child_second.shape


def test_k_point_crossover_correct():
    parent_first, parent_second = np.zeros((10, 10)), np.ones((10, 10))
    child_first, child_second = k_point_crossover(parent_first, parent_second, 'horizontal', k=4)

    print(f'Child first: {child_first}')
    print(f'Child second: {child_second}')

    assert parent_first.shape == child_first.shape == child_second.shape


def test_initial_pop_lhs_only_u_correct():
    source_shape = (10, 10)
    source_matrix = np.zeros(source_shape)
    pop_size = 10
    values_range = 10.0

    initial_pop = initial_pop_flat_lhs_only_u(pop_size=pop_size, bound_value=values_range, source_matrix=source_matrix)

    assert len(initial_pop) == pop_size
    assert initial_pop[0].genotype[0].shape == source_shape


def test_initial_population_only_u_rotations_correct():
    source_shape = (10, 10)
    source_matrix = np.zeros(source_shape)
    pop_size = 100
    max_radius = 10.0

    initial_pop = initial_population_only_u_rotations(pop_size=pop_size, source_matrix=source_matrix,
                                                      radius_range=max_radius)

    assert len(initial_pop) == pop_size
