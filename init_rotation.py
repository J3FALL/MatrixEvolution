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
    separate_crossover_only_u,
)
from evo_storage import EvoStorage


def evolution_only_u_component(source_matrix, crossover):
    mutation = partial(mutation_gauss, mu=0, sigma=0.3, prob_global=0.05)
    init_population = partial(initial_population_only_u_rotations, source_matrix=source_matrix,
                              radius_range=(0.0, 2.0), radius_ticks=5, axis=(2, 3))
    evo_operators = {'fitness': fitness_frob_norm_only_u,
                     'parent_selection': partial(select_by_tournament, tournament_size=20),
                     'mutation': partial(mutated_individ_only_u, mutate=mutation),
                     'crossover': partial(separate_crossover_only_u, crossover=crossover),
                     'initial_population': init_population}
    meta_params = {'pop_size': 100, 'generations': 1000, 'bound_value': 1.0,
                   'selection_rate': 0.1, 'crossover_rate': 0.70, 'random_selection_rate': 0.2, 'mutation_rate': 0.2}

    return evo_operators, meta_params


def evo_with_rotations(source_matrix, runs=10, crossover=partial(k_point_crossover, type='random', k=4), **kwargs):
    if 'storage_path' in kwargs:
        storage = EvoStorage(dump_file_path=kwargs['storage_path'], from_file=kwargs['storage_path'])
    else:
        storage = EvoStorage(dump_file_path='history.db')

    run_key = 'with_rot'
    if 'run_key' in kwargs:
        run_key = kwargs['run_key']

    evo_operators, meta_params = evolution_only_u_component(source_matrix=source_matrix, crossover=crossover)
    evo_history = EvoHistory(description=run_key)

    for run_id in range(runs):
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

    evo_history.fitness_history_boxplots(values_to_plot='min', save_to_file=False, gens_ticks=15)
    storage.save_run(key=run_key, evo_history=evo_history)


if __name__ == '__main__':
    source_matrix = np.random.rand(10, 10)
    evo_with_rotations(source_matrix=source_matrix, runs=5)
