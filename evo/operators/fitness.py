import numpy as np


def fitness_svd_frob_norm_only_u(source_matrix, genotype):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = genotype
    norm = np.linalg.norm(u_target - u_base)
    return norm


def fitness_svd_frob_norm(source_matrix, genotype):
    u, s, vh = genotype
    target = __matrix_from_svd(u=u, s=s, vh=vh)
    frob_norm = np.linalg.norm(source_matrix - target)
    return frob_norm


def fitness_s_component_diff(source_matrix, genotype):
    _, s_target, _ = genotype
    _, s_base, _ = np.linalg.svd(source_matrix, full_matrices=True)

    diff = np.sum(np.abs(np.sort(s_target) - np.sort(s_base)))
    return diff


def fitness_svd_eigen_values_norm(source_matrix, genotype):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = genotype
    eigen_values = np.linalg.eig(u_target - u_base)[0]
    norm = np.linalg.norm(eigen_values)
    return norm


def fitness_svd_inf_norm_only_u(source_matrix, genotype):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = genotype
    inf_metric = np.linalg.norm(u_target - u_base, ord=np.inf)

    return inf_metric


def fitness_svd_nuclear_norm_only_u(source_matrix, genotype):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = genotype
    nuclear_norm = np.linalg.norm(u_target - u_base, ord=2)

    return nuclear_norm


def fitness_svd_2_norm_only_u(source_matrix, genotype):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = genotype
    norm_2 = np.linalg.norm(u_target - u_base, ord=2)

    return norm_2


def fitness_svd_combined_norm_only_u(source_matrix, genotype):
    u_base, _, _ = np.linalg.svd(source_matrix, full_matrices=True)
    u_target, _, _ = genotype
    norm_2 = np.linalg.norm(u_target - u_base, ord=2)
    frob = np.linalg.norm(u_target - u_base)

    return frob + norm_2


def __matrix_from_svd(u, s, vh):
    return np.dot(u * s, vh)


def fitness_inverse_matrix_frob_norm(source_matrix, genotype):
    matrix_size = source_matrix.shape[0]
    target_identity = np.dot(source_matrix, genotype)

    frob = np.linalg.norm(np.identity(matrix_size) - target_identity)

    return frob
