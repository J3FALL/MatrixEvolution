from operator import attrgetter

import numpy as np


def fitness_frob_norm(source_matrix, svd):
    u, s, vh = svd
    target = __matrix_from_svd(u=u, s=s, vh=vh)
    frob_norm = np.linalg.norm(source_matrix - target)
    # second_ord_norm = np.linalg.norm(source_matrix - target, ord=2)
    return frob_norm


def new_individ_random_svd(source_matrix):
    size = source_matrix.shape
    u = random_matrix(size)
    s = 2 * np.random.rand(size[0], ) - 2
    vh = random_matrix(size)

    return u, s, vh


def new_individ_random_s_only(source_matrix):
    u_base, s_base, vh_base = np.linalg.svd(source_matrix, full_matrices=True)

    abs_range = 10
    s = abs_range * np.random.random(source_matrix.shape[0], ) - abs_range

    return u_base, s, vh_base


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


def mutation_gauss(candidate, mu, sigma, prob_global):
    source_shape = candidate.shape
    resulted = np.ndarray.flatten(candidate)
    for idx in range(len(resulted)):
        if np.random.random() < prob_global:
            resulted[idx] = np.random.normal(mu, sigma)
    return resulted.reshape(source_shape)


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


def random_matrix(matrix_size, range_value=2):
    size_a, size_b = matrix_size
    matrix = range_value * np.random.rand(size_a, size_b) - range_value

    return matrix


def __matrix_from_svd(u, s, vh):
    return np.dot(u * s, vh)
