import numpy as np
from pytest import raises

from evo.operators.init_population import (
    initial_pop_flat_lhs_only_u,
    initial_population_only_u_rotations,
    quadrant_position,
    initial_pop_only_u_fixed_quadrant
)
from utils import random_matrix


def test_initial_pop_lhs_only_u_correct():
    source_shape = (10, 10)
    source_matrix = np.zeros(source_shape)
    pop_size = 10
    values_range = 10.0

    initial_pop = initial_pop_flat_lhs_only_u(pop_size=pop_size,
                                              bound_value=values_range,
                                              source_matrix=source_matrix)

    assert len(initial_pop) == pop_size
    assert initial_pop[0].genotype[0].shape == source_shape


def test_initial_population_only_u_rotations_correct():
    source_shape = (10, 10)
    source_matrix = np.zeros(source_shape)
    pop_size = 100
    radius_range = (1.0, 5.0)

    initial_pop = initial_population_only_u_rotations(pop_size=pop_size,
                                                      source_matrix=source_matrix,
                                                      radius_range=radius_range)

    assert len(initial_pop) == pop_size


def test_initial_pop_only_u_fixed_quadrant_correct():
    matrix_size = (10, 10)
    source_matrix = random_matrix(matrix_size=matrix_size)
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)

    pop_size = 10
    quadrant_idx = 2
    i_from, i_to, j_from, j_to = quadrant_position(matrix_shape=matrix_size,
                                                   quadrant_idx=quadrant_idx)

    pop = initial_pop_only_u_fixed_quadrant(pop_size=pop_size,
                                            source_matrix=source_matrix,
                                            quadrant_idx=quadrant_idx)

    generated_u = pop[0].genotype[0]
    diff = np.abs(generated_u - u_base)
    zero_quadrant = np.zeros((matrix_size[0] // 2, matrix_size[1] // 2))
    actual_quadrant = diff[i_from:i_to, j_from:j_to]
    assert not np.array_equal(actual_quadrant, zero_quadrant)


def test_quadrant_position_correct():
    matrix_shape = (10, 10)
    quadrant_idx = 3
    expected_quadrant_position = (0, 5, 5, -1)

    actual_quadrant_position = quadrant_position(matrix_shape=matrix_shape,
                                                 quadrant_idx=quadrant_idx)

    assert expected_quadrant_position == actual_quadrant_position


def test_quadrant_position_exception():
    matrix_shape = (10, 10)
    invalid_quadrant_idx = -1

    with raises(Exception) as exc:
        assert quadrant_position(matrix_shape=matrix_shape,
                                 quadrant_idx=invalid_quadrant_idx)

    assert str(exc.value) == f'Unexpected quadrant_idx = {invalid_quadrant_idx}'
