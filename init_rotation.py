from functools import partial

import numpy as np

from evo_algorithm import (
    BasicEvoStrategy,
    EvoHistory
)
from evo_operators import (
    select_by_tournament,
    mutation_gauss,
    k_point_crossover,
    initial_population_only_u_rotations,
    fitness_frob_norm_only_u,
    mutated_individ_only_u,
    separate_crossover_only_u
)
from evo_storage import EvoStorage


def evolution_only_u_component(source_matrix):
    mutation = partial(mutation_gauss, mu=0, sigma=0.3, prob_global=0.05)
    crossover = partial(k_point_crossover, type='random', k=4)
    init_population = partial(initial_population_only_u_rotations, source_matrix=source_matrix,
                              radius_range=(0.0, 3.0), radius_ticks=10, axis=(3, 4))
    evo_operators = {'fitness': fitness_frob_norm_only_u,
                     'parent_selection': partial(select_by_tournament, tournament_size=20),
                     'mutation': partial(mutated_individ_only_u, mutate=mutation),
                     'crossover': partial(separate_crossover_only_u, crossover=crossover),
                     'initial_population': init_population}
    meta_params = {'pop_size': 100, 'generations': 1000, 'bound_value': 1.0,
                   'selection_rate': 0.2, 'crossover_rate': 0.80, 'random_selection_rate': 0.0, 'mutation_rate': 0.2}

    return evo_operators, meta_params


def evo_with_rotations(source_matrix):
    storage = EvoStorage(dump_file_path='history.db', from_file='history.db')
    evo_operators, meta_params = evolution_only_u_component(source_matrix=source_matrix)
    evo_history = EvoHistory(description='With rotations')

    for run_id in range(5):
        print(f'run_id: {run_id}')
        evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params,
                                        history=evo_history, source_matrix=source_matrix)
        evo_strategy.run()
        best_solution = evo_strategy.graded_by_fitness()[0]
        u_best = best_solution.genotype[0]
        print(u_best)
        u_baseline, _, _ = np.linalg.svd(source_matrix)
        print(u_baseline)
        print(np.abs(u_best - u_baseline))

    evo_history.fitness_history_boxplots(values_to_plot='min', save_to_file=False, gens_ticks=25)
    storage.save_run(key='with_rot', evo_history=evo_history)


if __name__ == '__main__':
    source_matrix = np.random.rand(10, 10)
    evo_with_rotations(source_matrix=source_matrix)
