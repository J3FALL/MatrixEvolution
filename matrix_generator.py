from functools import partial
from math import (
    cos, sin,
    radians)

import numpy as np
from scipy.optimize import (
    minimize)


def initial_diag_matrix(matrix_size, range_value):
    diag_vector = initial_diagonal_scaled(size=matrix_size, range_value=range_value)
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


def product_min(diag_values, range_value):
    return np.linalg.norm(range_value - np.prod(diag_values))


def initial_diagonal_minimized(matrix_size, range_value, shuffle=True):
    initial_values = np.random.randn(matrix_size, 1)

    result = minimize(partial(product_min, range_value=range_value), initial_values,
                      method='SLSQP',
                      options={'disp': False})
    diag_values = result.x
    if shuffle:
        np.random.shuffle(diag_values)
    matrix = np.diag(diag_values)
    return matrix


def initial_diagonal_scaled(size, range_value):
    scaled_value = np.random.randint(int(10e6), 5 * int(10e6))

    int_parts = random_integers(amount=int(size / 2))
    frac_parts = []

    while (len(int_parts) + len(frac_parts)) < size:
        value = int_parts.pop(int_parts.index(min(int_parts)))

        frac_parts.append(1.0 / value)
        int_parts.append(value * value)
    prod = np.prod(np.asarray(int_parts + frac_parts))
    scale = prod / range_value

    resulted = np.asarray(int_parts + frac_parts)
    resulted = resulted / scale

    np.random.shuffle(resulted)
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


def random_integers(amount=10):
    values_range = np.arange(1, 100)
    values_range = values_range[values_range != 0]

    values = list(np.random.choice(values_range, amount))
    return values
