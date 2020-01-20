import random

import numpy as np

from matrix import MatrixIndivid
from utils import quadrant_position


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


def geo_crossover_fixed_box(parent_first, parent_second, box_size, **kwargs):
    size = parent_first.shape
    top_left = np.random.randint(low=0, high=size[0]), np.random.randint(low=0, high=size[1])

    child_first, child_second = geo_crossover(parent_first=parent_first, parent_second=parent_second, random_box=False,
                                              top_left=top_left, box_size=box_size)

    return child_first, child_second


# TODO: change box_size to be adaptive to total_generations
def decreasing_dynamic_geo_crossover(parent_first, parent_second, box_size_initial, current_gen, **kwargs):
    box_size = box_size_initial - current_gen // 200
    return geo_crossover_fixed_box(parent_first=parent_first, parent_second=parent_second, box_size=box_size, **kwargs)


def increasing_dynamic_geo_crossover(parent_first, parent_second, box_size_initial, current_gen, **kwargs):
    box_size = box_size_initial + current_gen // 200
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


def crossover_inverse(parent_first: MatrixIndivid, parent_second: MatrixIndivid,
                      crossover, **kwargs):
    inv_first, inv_second = crossover(parent_first.genotype, parent_second.genotype,
                                      **kwargs)
    child_first = MatrixIndivid(genotype=inv_first)
    child_second = MatrixIndivid(genotype=inv_second)

    return child_first, child_second


def swap_crossover(parent_first, parent_second, rows_to_swap=2, **kwargs):
    matrix_size = parent_first.shape[0]
    rows_idxs = np.random.choice(np.arange(0, matrix_size), size=rows_to_swap,
                                 replace=False)

    child_first, child_second = np.copy(parent_first), np.copy(parent_second)

    for idx in rows_idxs:
        child_first[idx, :] = parent_second[idx, :]
        child_second[idx, :] = parent_first[idx, :]

    return child_first, child_second
