import numpy as np


def initial_diag_matrix(size, norm_value):
    norm_value = np.round(norm_value)

    int_parts = prime_factors(norm_value)
    frac_parts = []

    while (len(int_parts) + len(frac_parts)) < size:
        value = int_parts.pop(int_parts.index(min(int_parts)))
        frac_parts.append(1.0 / value)
        int_parts.append(value * value)

    resulted = np.asarray(int_parts + frac_parts)
    return resulted


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
