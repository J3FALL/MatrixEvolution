from math import (
    cos, sin,
    radians)

import numpy as np


def initial_diag_matrix(matrix_size, norm_value):
    diag_vector = initial_diagonal_by_prime_fraction(size=matrix_size, norm_value=norm_value)
    matrix = np.diag(diag_vector)

    return matrix


def rotate_matrix(source_matrix, axis, angle):
    size = source_matrix.shape[0]
    rot_matrix = rotation_matrix(size=size, axis=axis, angle=angle)
    resulted = np.dot(rot_matrix, source_matrix)
    return resulted


def rotation_matrix(size, axis, angle):
    i, j = axis
    angle_rad = radians(angle)

    rot_matrix = np.diag(np.ones((size,)))

    rot_matrix[i, i] = cos(angle_rad)
    rot_matrix[j, j] = cos(angle_rad)
    rot_matrix[i, j] = (-1.0) * sin(angle_rad)
    rot_matrix[j, i] = sin(angle_rad)

    return rot_matrix


def initial_diagonal_by_prime_fraction(size, norm_value):
    norm_value = np.round(norm_value)

    int_parts = prime_factors(norm_value)
    frac_parts = []

    while (len(int_parts) + len(frac_parts)) < size:
        value = int_parts.pop(int_parts.index(min(int_parts)))
        frac_parts.append(1.0 / value)
        int_parts.append(value * value)

    resulted = np.asarray(int_parts + frac_parts)
    return resulted


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
