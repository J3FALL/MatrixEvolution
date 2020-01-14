from math import (
    cos,
    radians)

import numpy as np

from matrix_generator import (
    initial_diagonal_scaled,
    initial_diag_matrix,
    rotation_matrix
)


def test_initial_diag_matrix_correct():
    matrix_size, norm_value = 10, 0.5

    matrix = initial_diag_matrix(matrix_size=matrix_size,
                                 range_value=norm_value)

    assert matrix.shape == (matrix_size, matrix_size)


def test_initial_diagonal_by_prime_fraction_correct():
    diag_size, norm_value = 2, 10
    resulted_matrix = initial_diagonal_scaled(size=diag_size,
                                              range_value=norm_value)

    assert resulted_matrix.shape == (diag_size,)


def test_initial_diagonal_frac_part_correct():
    diag_size, norm_value = 10, 10
    resulted_matrix = initial_diagonal_scaled(size=diag_size,
                                              range_value=norm_value)

    assert resulted_matrix.shape == (diag_size,)


def test_rotation_matrix_correct():
    size, axis, angle = 10, (2, 1), 30
    rot_matrix = rotation_matrix(size=size, axis=axis, angle=angle)

    assert rot_matrix.shape == (size, size)
    assert np.allclose(rot_matrix[1, 1], cos(radians(angle)), 10e-8)
