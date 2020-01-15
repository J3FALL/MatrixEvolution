import numpy as np

from evo.operators.crossover import (
    single_point_crossover,
    two_point_crossover,
    k_point_crossover,
    geo_crossover
)
from utils import random_matrix


def test_single_point_crossover_correct():
    parent_first, parent_second = random_matrix(matrix_size=(3, 3)), random_matrix(matrix_size=(3, 3))

    child_first, child_second = single_point_crossover(parent_first, parent_second, 'vertical')

    print(f'Parent first: {parent_first}')
    print(f'Parent second: {parent_second}')
    print(f'Child first: {child_first}')
    print(f'Child second{child_second}')

    assert parent_first.shape == child_first.shape == child_second.shape


def test_two_point_crossover_correct():
    parent_first, parent_second = np.zeros((5, 5)), np.ones((5, 5))

    child_first, child_second = two_point_crossover(parent_first, parent_second, 'horizontal')

    print(f'Child first: {child_first}')
    print(f'Child second: {child_second}')

    assert parent_first.shape == child_first.shape == child_second.shape


def test_k_point_crossover_correct():
    parent_first, parent_second = np.zeros((10, 10)), np.ones((10, 10))
    child_first, child_second = k_point_crossover(parent_first, parent_second, 'horizontal', k=4)

    print(f'Child first: {child_first}')
    print(f'Child second: {child_second}')

    assert parent_first.shape == child_first.shape == child_second.shape


def test_geo_crossover_random_box_correct():
    shape = (10, 10)
    parent_first, parent_second = np.zeros(shape), np.ones(shape)

    child_first, child_second = geo_crossover(parent_first=parent_first, parent_second=parent_second)

    assert child_first.shape == child_second.shape == parent_first.shape


def test_geo_crossover_fixed_box_correct():
    shape = (10, 10)
    parent_first, parent_second = np.zeros(shape), np.ones(shape)
    top_left = (3, 3)
    box_size = 3
    child_first, child_second = geo_crossover(parent_first=parent_first, parent_second=parent_second,
                                              random_box=False, top_left=top_left, box_size=box_size)

    expected_box = parent_first[top_left[0]:top_left[0] + box_size, top_left[1]: top_left[1] + box_size]
    actual_box = child_first[top_left[0]:top_left[0] + box_size, top_left[1]: top_left[1] + box_size]

    assert np.array_equal(expected_box, actual_box)
