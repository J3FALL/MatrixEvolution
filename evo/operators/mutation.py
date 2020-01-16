import math

import numpy as np

from matrix import MatrixIndivid
from utils import quadrant_position


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


def mutated_individ(source_individ):
    u_mutated = mutation_gauss(candidate=source_individ.genotype[0], mu=0, sigma=0.2, prob_global=0.05)
    s_mutated = mutation_gauss(candidate=source_individ.genotype[1], mu=0, sigma=0.2, prob_global=0.05)
    vh_mutated = mutation_gauss(candidate=source_individ.genotype[2], mu=0, sigma=0.2, prob_global=0.05)

    resulted = MatrixIndivid(genotype=(u_mutated, s_mutated, vh_mutated))

    return resulted


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


def mutated_individ_inverse(source_individ: MatrixIndivid, mutate):
    inv_matrix = source_individ.genotype
    inv_mutated = mutate(candidate=inv_matrix)
    resulted = MatrixIndivid(genotype=inv_mutated)

    return resulted
