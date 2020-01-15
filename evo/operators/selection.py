from operator import attrgetter

import numpy as np


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
