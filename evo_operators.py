import math
import random
from operator import attrgetter

import numpy as np
from pysampling.sample import sample

from matrix import MatrixIndivid
from matrix_generator import (
    rotate_matrix,
    initial_diagonal_minimized
)


# TODO: script is too heavy, new package with operators as separate scripts is required

def fitness_frob_norm(source_matrix, svd):
    u, s, vh = svd
    target = __matrix_from_svd(u=u, s=s, vh=vh)
    frob_norm = np.linalg.norm(source_matrix - target)
    return frob_norm


def fitness_s_component_diff(source_matrix, svd):
    _, s_target, _ = svd
    _, s_base, _ = np.linalg.svd(source_matrix, full_matrices=True)

    diff = np.sum(np.abs(np.sort(s_target) - np.sort(s_base)))
    return diff


def fitness_frob_norm_only_u(source_matrix, svd):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = svd
    norm = np.linalg.norm(u_target - u_base)
    return norm


def fitness_eigen_values_norm(source_matrix, svd):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = svd
    eigen_values = np.linalg.eig(u_target - u_base)[0]
    norm = np.linalg.norm(eigen_values)
    return norm


def fitness_inf_norm_only_u(source_matrix, svd):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = svd
    inf_metric = np.linalg.norm(u_target - u_base, ord=np.inf)

    return inf_metric


def fitness_nuclear_norm_only_u(source_matrix, svd):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = svd
    nuclear_norm = np.linalg.norm(u_target - u_base, ord=2)

    return nuclear_norm


def fitness_2_norm_only_u(source_matrix, svd):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = svd
    norm_2 = np.linalg.norm(u_target - u_base, ord=2)

    return norm_2


def fitness_combined_norm_only_u(source_matrix, svd):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = svd
    norm_2 = np.linalg.norm(u_target - u_base, ord=2)
    frob = np.linalg.norm(u_target - u_base)

    return frob + norm_2


def new_individ_random_svd(source_matrix, bound_value=10.0):
    size = source_matrix.shape
    u = random_matrix(size, bound_value=bound_value)
    s = bound_value * np.random.rand(size[0], ) - bound_value / 2.0
    vh = random_matrix(size, bound_value=bound_value)
    return u, s, vh


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


def mutated_individ_only_s(source_individ: MatrixIndivid, mutate):
    u, s, vh = source_individ.genotype
    u_resulted = np.copy(u)
    vh_resulted = np.copy(vh)

    s_mutated = mutate(candidate=s)
    resulted = MatrixIndivid(genotype=(u_resulted, s_mutated, vh_resulted))

    return resulted


def mutated_individ_only_u(source_individ: MatrixIndivid, mutate):
    u, s, vh = source_individ.genotype
    s_resulted = np.copy(s)
    vh_resulted = np.copy(vh)

    u_mutated = mutate(candidate=u)
    resulted = MatrixIndivid(genotype=(u_mutated, s_resulted, vh_resulted))

    return resulted


def separate_crossover_only_s(parent_first: MatrixIndivid, parent_second: MatrixIndivid, crossover):
    u_first, u_second = parent_first.genotype[0], parent_second.genotype[0],

    s_first, s_second = crossover(parent_first=parent_first.genotype[1],
                                  parent_second=parent_second.genotype[1])

    vh_first, vh_second = parent_first.genotype[2], parent_second.genotype[2]

    child_first = MatrixIndivid(genotype=(u_first, s_first, vh_first))
    child_second = MatrixIndivid(genotype=(u_second, s_second, vh_second))

    return child_first, child_second


def separate_crossover_only_u(parent_first: MatrixIndivid, parent_second: MatrixIndivid, crossover, **kwargs):
    u_first, u_second = crossover(parent_first.genotype[0], parent_second.genotype[0], **kwargs)

    s_first, s_second = parent_first.genotype[1], parent_second.genotype[1]

    vh_first, vh_second = parent_first.genotype[2], parent_second.genotype[2]

    child_first = MatrixIndivid(genotype=(u_first, s_first, vh_first))
    child_second = MatrixIndivid(genotype=(u_second, s_second, vh_second))

    return child_first, child_second


def mutated_individ(source_individ):
    u_mutated = mutation_gauss(candidate=source_individ.genotype[0], mu=0, sigma=0.2, prob_global=0.05)
    s_mutated = mutation_gauss(candidate=source_individ.genotype[1], mu=0, sigma=0.2, prob_global=0.05)
    vh_mutated = mutation_gauss(candidate=source_individ.genotype[2], mu=0, sigma=0.2, prob_global=0.05)

    resulted = MatrixIndivid(genotype=(u_mutated, s_mutated, vh_mutated))

    return resulted


def single_point_crossover(parent_first, parent_second, type='horizontal', **kwargs):
    size = parent_first.shape

    child_first, child_second = np.zeros(shape=size), np.zeros(shape=size)

    if type == 'horizontal':
        cross_point = np.random.randint(0, size[0])
        if 'cross_point' in kwargs:
            cross_point = kwargs['cross_point']

        child_first[:cross_point] = parent_first[:cross_point]
        child_first[cross_point:] = parent_second[cross_point:]

        child_second[:cross_point] = parent_second[:cross_point]
        child_second[cross_point:] = parent_first[cross_point:]
    elif type == 'vertical':
        cross_point = np.random.randint(0, size[1])
        if 'cross_point' in kwargs:
            cross_point = kwargs['cross_point']
        child_first[:, :cross_point] = parent_first[:, :cross_point]
        child_first[:, cross_point:] = parent_second[:, cross_point:]

        child_second[:, :cross_point] = parent_second[:, :cross_point]
        child_second[:, cross_point:] = parent_first[:, cross_point:]

    return child_first, child_second


def two_point_crossover(parent_first, parent_second, type='horizontal'):
    size = parent_first.shape

    child_first, child_second = np.copy(parent_first), np.copy(parent_second)

    if type == 'horizontal':
        cross_point_first = np.random.randint(0, size[0])
        cross_point_second = np.random.randint(0, size[0] - 1)
        if cross_point_first >= cross_point_second:
            cross_point_first, cross_point_second = cross_point_second, cross_point_first
        child_first[cross_point_first:cross_point_second] = parent_second[cross_point_first:cross_point_second]
        child_second[cross_point_first:cross_point_second] = parent_first[cross_point_first:cross_point_second]

    elif type == 'vertical':
        cross_point_first = np.random.randint(0, size[1])
        cross_point_second = np.random.randint(0, size[1] - 1)
        if cross_point_first >= cross_point_second:
            cross_point_first, cross_point_second = cross_point_second, cross_point_first
        child_first[:, cross_point_first:cross_point_second] = parent_second[:, cross_point_first:cross_point_second]
        child_second[:, cross_point_first:cross_point_second] = parent_first[:, cross_point_first:cross_point_second]

    return child_first, child_second


def k_point_crossover(parent_first, parent_second, type='horizontal', k=3, **kwargs):
    size = parent_first.shape
    child_first, child_second = np.zeros(size), np.zeros(size)

    if type == 'random':
        type = np.random.choice(['horizontal', 'vertical'])

    if type == 'horizontal':
        points = __random_cross_points(max_size=size[0], k=k)
        parents = [parent_first, parent_second]
        parent_idx = 0

        for point_idx in range(1, len(points)):
            point_from, point_to = points[point_idx - 1], points[point_idx]

            child_first[point_from: point_to] = parents[parent_idx][point_from:point_to]
            child_second[point_from:point_to] = parents[(parent_idx + 1) % 2][point_from:point_to]

            parent_idx = (parent_idx + 1) % 2
    elif type == 'vertical':
        points = __random_cross_points(max_size=size[1], k=k)

        parents = [parent_first, parent_second]
        parent_idx = 0

        for point_idx in range(1, len(points)):
            point_from, point_to = points[point_idx - 1], points[point_idx]

            child_first[:, point_from: point_to] = parents[parent_idx][:, point_from:point_to]
            child_second[:, point_from:point_to] = parents[(parent_idx + 1) % 2][:, point_from:point_to]

            parent_idx = (parent_idx + 1) % 2

    return child_first, child_second


def geo_crossover(parent_first, parent_second, random_box=True, **kwargs):
    size = parent_first.shape

    if random_box:
        top_left = (np.random.randint(low=0, high=size[0]),
                    np.random.randint(low=0, high=size[1]))
        box_size = np.random.randint(low=0, high=size[0])
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)
    else:
        box_size = kwargs['box_size']
        top_left = kwargs['top_left']
        bottom_right = (top_left[0] + box_size, top_left[1] + box_size)

    inside_mask = np.zeros(size)
    inside_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1.0

    child_first = inside_mask * parent_first + (1.0 - inside_mask) * parent_second
    child_second = inside_mask * parent_second + (1.0 - inside_mask) * parent_first

    return child_first, child_second


def __random_cross_points(max_size, k=3):
    points = random.sample(range(0, max_size), k)
    if 0 not in points:
        points.append(0)
    if max_size not in points:
        points.append(max_size)
    points = sorted(points)

    return points


def arithmetic_crossover(parent_first, parent_second, **kwargs):
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
    else:
        alpha = np.random.random()

    child_first = alpha * parent_first + (1.0 - alpha) * parent_second
    child_second = (1.0 - alpha) * parent_first + alpha * parent_second

    return child_first, child_second


def joint_crossover(parent_first, parent_second):
    # TODO: refactor this
    crossover_type = np.random.choice(['horizontal', 'vertical'])

    min_size = np.min(parent_first.genotype[0].shape)
    cross_point = np.random.randint(0, min_size - 1)
    u_first, u_second = single_point_crossover(parent_first=parent_first.genotype[0],
                                               parent_second=parent_second.genotype[0],
                                               type=crossover_type, cross_point=cross_point)
    s_first, s_second = single_point_crossover(parent_first=parent_first.genotype[1],
                                               parent_second=parent_second.genotype[1],
                                               type='horizontal', cross_point=cross_point)
    vh_first, vh_second = single_point_crossover(parent_first=parent_first.genotype[2],
                                                 parent_second=parent_second.genotype[2],
                                                 type=crossover_type, cross_point=cross_point)

    child_first = MatrixIndivid(genotype=(u_first, s_first, vh_first))
    child_second = MatrixIndivid(genotype=(u_second, s_second, vh_second))

    return child_first, child_second


def separate_crossover(parent_first, parent_second):
    crossover_type = np.random.choice(['horizontal', 'vertical'])
    u_first, u_second = k_point_crossover(parent_first=parent_first.genotype[0],
                                          parent_second=parent_second.genotype[0],
                                          type=crossover_type, k=4)
    s_first, s_second = k_point_crossover(parent_first=parent_first.genotype[1],
                                          parent_second=parent_second.genotype[1],
                                          type='horizontal', k=4)
    vh_first, vh_second = k_point_crossover(parent_first=parent_first.genotype[2],
                                            parent_second=parent_second.genotype[2],
                                            type=crossover_type, k=4)

    child_first = MatrixIndivid(genotype=(u_first, s_first, vh_first))
    child_second = MatrixIndivid(genotype=(u_second, s_second, vh_second))

    return child_first, child_second


def mutation_gauss(candidate, mu, sigma, prob_global):
    source_shape = candidate.shape
    resulted = np.ndarray.flatten(candidate)

    chosen_values_amount = math.ceil(prob_global * len(resulted))
    idx_to_mutate = np.random.choice(np.arange(0, len(resulted)), chosen_values_amount, replace=False)
    for idx in idx_to_mutate:
        resulted[idx] = np.random.normal(mu, sigma)

    return resulted.reshape(source_shape)


def mutation_quadrant_gauss(candidate, quadrant_idx, mu, sigma, prob_global):
    i_from, i_to, j_from, j_to = quadrant_position(matrix_shape=candidate.shape,
                                                   quadrant_idx=quadrant_idx)

    quad_candidate_first = candidate[i_from:i_to, j_from:j_to]
    mutated_quad = mutation_gauss(quad_candidate_first,
                                  mu, sigma, prob_global)
    resulted = np.copy(candidate)
    resulted[i_from:i_to, j_from:j_to] = mutated_quad

    return resulted


def select_k_best(candidates, k):
    assert k <= len(candidates)

    graded = sorted(candidates, key=attrgetter('fitness_value'))

    return graded[:k]


def select_by_tournament(candidates, k, tournament_size=10):
    chosen = []
    for _ in range(k):
        aspirants = np.random.choice(candidates, tournament_size)
        chosen.append(min(aspirants, key=attrgetter('fitness_value')))

    return chosen


def select_by_rank(candidates, k):
    candidates_amount = len(candidates)
    rank_sum = candidates_amount * (candidates_amount + 1) / 2
    graded = sorted(candidates, key=attrgetter('fitness_value'), reverse=True)

    selected = []

    while len(selected) < k:
        for rank, candidate in enumerate(graded, 1):
            select_prob = float(rank) / rank_sum
            if np.random.random() < select_prob:
                selected.append(candidate)

    return selected[:k]


def random_matrix(matrix_size, bound_value=1.0):
    size_a, size_b = matrix_size
    matrix = bound_value * np.random.rand(size_a, size_b) - bound_value / 2.0

    return matrix


def __matrix_from_svd(u, s, vh):
    return np.dot(u * s, vh)


def geo_crossover_fixed_box(parent_first, parent_second, box_size, **kwargs):
    size = parent_first.shape
    top_left = np.random.randint(low=0, high=size[0]), np.random.randint(low=0, high=size[1])

    child_first, child_second = geo_crossover(parent_first=parent_first, parent_second=parent_second, random_box=False,
                                              top_left=top_left, box_size=box_size)

    return child_first, child_second


# TODO: change box_size to be adaptive to total_generations
def decreasing_dynamic_geo_crossover(parent_first, parent_second, box_size_initial, current_gen, **kwargs):
    box_size = box_size_initial - current_gen // 100
    return geo_crossover_fixed_box(parent_first=parent_first, parent_second=parent_second, box_size=box_size, **kwargs)


def increasing_dynamic_geo_crossover(parent_first, parent_second, box_size_initial, current_gen, **kwargs):
    box_size = box_size_initial + current_gen // 100
    return geo_crossover_fixed_box(parent_first=parent_first, parent_second=parent_second, box_size=box_size, **kwargs)


def quadrant_increasing_dynamic_geo_crossover(parent_first, parent_second,
                                              box_size_initial, current_gen,
                                              quadrant_idx, **kwargs):
    i_from, i_to, j_from, j_to = quadrant_position(matrix_shape=parent_first.shape,
                                                   quadrant_idx=quadrant_idx)

    quad_parent_first = parent_first[i_from:i_to, j_from:j_to]
    quad_parent_second = parent_second[i_from:i_to, j_from:j_to]

    box_size = box_size_initial + current_gen // 300
    quad_child_first, quad_child_second = geo_crossover_fixed_box(parent_first=quad_parent_first,
                                                                  parent_second=quad_parent_second,
                                                                  box_size=box_size, **kwargs)

    child_first, child_second = np.copy(parent_first), np.copy(parent_second)

    child_first[i_from:i_to, j_from:j_to] = quad_child_first
    child_second[i_from:i_to, j_from:j_to] = quad_child_second

    return child_first, child_second
