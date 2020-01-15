import numpy as np
from pysampling.sample import sample

from matrix import MatrixIndivid
from matrix_generator import (
    rotate_matrix,
    initial_diagonal_minimized
)
from utils import quadrant_position


def new_individ_random_svd(source_matrix, bound_value=10.0):
    size = source_matrix.shape
    u = random_matrix(size, bound_value=bound_value)
    s = bound_value * np.random.rand(size[0], ) - bound_value / 2.0
    vh = random_matrix(size, bound_value=bound_value)
    return u, s, vh


def random_matrix(matrix_size, bound_value=1.0):
    size_a, size_b = matrix_size
    matrix = bound_value * np.random.rand(size_a, size_b) - bound_value / 2.0

    return matrix


def new_individ_random_s_only(source_matrix):
    u_base, s_base, vh_base = np.linalg.svd(source_matrix, full_matrices=True)

    abs_range = 10.0
    s = abs_range * np.random.random(source_matrix.shape[0], ) - abs_range / 2.0

    return u_base, s, vh_base


def init_population_random(pop_size, source_matrix, bound_value=10.0):
    pop = []
    for _ in range(pop_size):
        individ = MatrixIndivid(genotype=new_individ_random_svd(source_matrix, bound_value=bound_value))
        pop.append(individ)
    return pop


def initial_population_from_lhs_only_s(samples_amount, vector_size, values_range, source_matrix):
    print('Sampling from LHS...')
    s_samples = values_range * sample("lhs", samples_amount, vector_size) - values_range / 2.0
    print('Sampling: done')
    u_base, _, vh_base = np.linalg.svd(source_matrix, full_matrices=True)

    pop = []
    for idx, s in enumerate(s_samples):
        pop.append(MatrixIndivid(genotype=(u_base, s, vh_base)))
    return pop


def initial_population_only_u_random(pop_size, source_matrix, bound_value=10.0):
    _, s_base, vh_base = np.linalg.svd(source_matrix, full_matrices=True)
    size = source_matrix.shape

    pop = []
    for _ in range(pop_size):
        u = random_matrix(size, bound_value=bound_value)
        pop.append(MatrixIndivid(genotype=(u, s_base, vh_base)))

    return pop


def initial_pop_only_u_fixed_quadrant(pop_size, source_matrix, bound_value=10.0, quadrant_idx=1):
    u_base, s_base, vh_base = np.linalg.svd(source_matrix, full_matrices=True)
    size = source_matrix.shape

    pop = []
    for _ in range(pop_size):
        random_u = random_matrix(size, bound_value=bound_value)
        i_from, i_to, j_from, j_to = quadrant_position(matrix_shape=size, quadrant_idx=quadrant_idx)
        modified_u = u_base
        modified_u[i_from:i_to, j_from:j_to] = random_u[i_from:i_to, j_from:j_to]
        pop.append(MatrixIndivid(genotype=(modified_u, s_base, vh_base)))

    return pop


def initial_pop_flat_lhs_only_u(pop_size, source_matrix, bound_value):
    source_shape = source_matrix.shape
    vector_size = np.ndarray.flatten(source_matrix).shape[0]

    _, s_base, vh_base = np.linalg.svd(source_matrix, full_matrices=True)

    print('Sampling from LHS...')
    u_samples = bound_value * sample("lhs", pop_size, vector_size) - bound_value / 2.0
    print('Sampling: done')

    pop = []
    for idx, u in enumerate(u_samples):
        pop.append(MatrixIndivid(genotype=(u.reshape(source_shape), s_base, vh_base)))

    return pop


def initial_population_only_u_rotations(pop_size, source_matrix, radius_range=(0.1, 0.5), radius_ticks=5, axis=(0, 1)):
    _, s_base, vh_base = np.linalg.svd(source_matrix, full_matrices=True)
    size = source_matrix.shape[0]
    pop = []
    for radius in np.linspace(radius_range[0], radius_range[1], radius_ticks):
        points_amount = int(pop_size / radius_ticks)
        u_diag = initial_diagonal_minimized(matrix_size=size, range_value=radius)
        for k in range(points_amount):
            angle = 360.0 / points_amount * k
            u_resulted = rotate_matrix(source_matrix=u_diag, axis=axis, angle=angle)
            pop.append(MatrixIndivid(genotype=(u_resulted, s_base, vh_base)))

    return pop
