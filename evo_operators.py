from operator import attrgetter

import numpy as np


def fitness_frob_norm(source_matrix, svd):
    u, s, vh = svd
    target = __matrix_from_svd(u=u, s=s, vh=vh)
    frob_norm = np.linalg.norm(source_matrix - target)

    return frob_norm


def new_individ_random_svd(source_matrix_size):
    u = random_matrix(source_matrix_size)
    s = np.random.rand(source_matrix_size[0], )
    vh = random_matrix(source_matrix_size)

    return u, s, vh


def single_point_crossover(parent_first, parent_second, type='horizontal', **kwargs):
    size = parent_first.shape

    child_first, child_second = np.zeros(shape=size), np.zeros(shape=size)

    if type == 'horizontal':
        cross_point = np.random.randint(0, size[0] - 1)
        if 'cross_point' in kwargs:
            cross_point = kwargs['cross_point']

        child_first[:cross_point] = parent_first[:cross_point]
        child_first[cross_point:] = parent_second[cross_point:]

        child_second[:cross_point] = parent_second[:cross_point]
        child_second[cross_point:] = parent_first[cross_point:]
    elif type == 'vertical':
        cross_point = np.random.randint(0, size[1] - 1)
        if 'cross_point' in kwargs:
            cross_point = kwargs['cross_point']
        child_first[:, :cross_point] = parent_first[:, :cross_point]
        child_first[:, cross_point:] = parent_second[:, cross_point:]

        child_second[:, :cross_point] = parent_second[:, :cross_point]
        child_second[:, cross_point:] = parent_first[:, cross_point:]

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


def random_matrix(matrix_size):
    size_a, size_b = matrix_size
    matrix = np.random.rand(size_a, size_b)

    return matrix


def __matrix_from_svd(u, s, vh):
    return np.dot(u * s, vh)
