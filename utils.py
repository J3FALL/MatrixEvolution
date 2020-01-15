import numpy as np


def random_matrix(matrix_size, bound_value=1.0):
    size_a, size_b = matrix_size
    matrix = bound_value * np.random.rand(size_a, size_b) - bound_value / 2.0

    return matrix


def quadrant_position(matrix_shape, quadrant_idx):
    quadrant_functions = {
        1: lambda idx: (matrix_shape[0] // 2, -1, 0, matrix_shape[1] // 2),
        2: lambda idx: (0, matrix_shape[0] // 2, 0, matrix_shape[1] // 2),
        3: lambda idx: (0, matrix_shape[0] // 2, matrix_shape[1] // 2, -1),
        4: lambda idx: (matrix_shape[0] // 2, -1, matrix_shape[1] // 2, -1)
    }

    if quadrant_idx not in quadrant_functions.keys():
        raise Exception(f'Unexpected quadrant_idx = {quadrant_idx}')

    return quadrant_functions[quadrant_idx](quadrant_idx)
