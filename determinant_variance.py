from functools import partial

import numpy as np
from scipy.optimize import (
    minimize)

from evo.operators.init_population import (
    initial_population_only_u_random,
    initial_population_only_u_rotations
)
from matrix_generator import initial_diagonal_minimized


def init_pop_randomly():
    source_matrix = np.random.rand(20, 20)
    pop_size = 100

    pop = initial_population_only_u_random(pop_size=pop_size,
                                           source_matrix=source_matrix,
                                           bound_value=1.1)

    u_values = [individ.genotype[0] for individ in pop]

    determinant_variance = np.var([np.linalg.det(u) for u in u_values])
    determinant_mean = np.mean([np.linalg.det(u) for u in u_values])
    print(f'Det: {determinant_mean} +/- {determinant_variance}')


def init_pop_with_rotation():
    source_matrix = np.random.rand(20, 20)
    pop_size, radius_range = 50, (0.1, 3.0)
    radius_ticks, axis = 5, (4, 5)

    pop = initial_population_only_u_rotations(pop_size=pop_size,
                                              source_matrix=source_matrix,
                                              radius_range=radius_range,
                                              radius_ticks=radius_ticks,
                                              axis=axis)

    u_values = [individ.genotype[0] for individ in pop]

    determinant_variance = np.var([np.linalg.det(u) for u in u_values])
    determinant_mean = np.mean([np.linalg.det(u) for u in u_values])
    print(f'Det: {determinant_mean} +/- {determinant_variance}')


def product_min(diag_values, range_value):
    return np.linalg.norm(range_value - np.prod(diag_values))


def find_min_of_product():
    range_value = 0.5
    initial_values = np.random.randn(10, 1)
    print(product_min(diag_values=initial_values, range_value=range_value))

    result = minimize(partial(product_min, range_value=range_value), initial_values,
                      method='SLSQP',
                      options={'disp': True})
    print(result.x)
    print(product_min(result.x, range_value))


def initial_diag_variance_discovery():
    size, range_value = 10, 0.5
    samples_amount = 1000
    samples = np.zeros((samples_amount, size))

    for idx in range(samples_amount):
        matrix = initial_diagonal_minimized(matrix_size=size, range_value=range_value)
        diag = matrix.diagonal()
        samples[idx, :] = diag

    for val_idx in range(size):
        mean, var = np.mean(samples[:, val_idx]), np.var(samples[:, val_idx])
        print(f'Value idx {val_idx}: {mean} +/- {var}')


if __name__ == '__main__':
    init_pop_with_rotation()
