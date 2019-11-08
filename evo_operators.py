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


def single_point_crossover(parent_first, parent_second, type='horizontal'):
    size = parent_first.shape

    child_first, child_second = np.zeros(shape=size), np.zeros(shape=size)

    if type is 'horizontal':
        cross_point = np.random.randint(0, size[0] - 1)

        child_first[:cross_point] = parent_first[:cross_point]
        child_first[cross_point:] = parent_second[cross_point:]

        child_second[:cross_point] = parent_second[:cross_point]
        child_second[cross_point:] = parent_first[cross_point:]
    elif type is 'vertical':
        cross_point = np.random.randint(0, size[1] - 1)

        child_first[:, :cross_point] = parent_first[:, :cross_point]
        child_first[:, cross_point:] = parent_second[:, cross_point:]

        child_second[:, :cross_point] = parent_second[:, :cross_point]
        child_second[:, cross_point:] = parent_first[:, cross_point:]

    return child_first, child_second


def random_matrix(matrix_size):
    size_a, size_b = matrix_size
    matrix = np.random.rand(size_a, size_b)

    return matrix


def __matrix_from_svd(u, s, vh):
    return np.dot(u * s, vh)
