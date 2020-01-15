import math

import numpy as np

from evo.operators.mutation import mutation_gauss
from utils import random_matrix


def test_mutation_gauss_correct():
    candidate = random_matrix(matrix_size=(10, 10))
    source_shape = candidate.shape
    mu, sigma = 0, 0.1
    prob_global = 0.05
    resulted = mutation_gauss(candidate=candidate, mu=mu, sigma=sigma, prob_global=prob_global)

    assert source_shape == resulted.shape


def test_mutation_gauss_probability_correct():
    matrix_size = (10, 10)
    candidate = random_matrix(matrix_size=matrix_size)
    mu, sigma = 0.1, 0.05
    prob_global = 0.1
    resulted = mutation_gauss(candidate=candidate, mu=mu, sigma=sigma, prob_global=prob_global)

    diff_matrix = np.abs(resulted - candidate)

    expected_mutations = math.ceil(prob_global * matrix_size[0] * matrix_size[1])
    actual_mutations = (diff_matrix > 0.0).sum()
    assert expected_mutations == actual_mutations
