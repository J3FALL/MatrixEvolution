from utils import random_matrix


def test_new_individ_random_correct():
    expected_size = (10, 10)
    matrix = random_matrix(matrix_size=expected_size)
    actual_size = matrix.shape

    assert expected_size == actual_size
