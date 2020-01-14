from functools import partial

import numpy as np

from evo_algorithm import (
    BasicEvoStrategy,
    EvoHistory
)
from evo_operators import (
    quadrant_increasing_dynamic_geo_crossover,
    initial_pop_only_u_fixed_quadrant,
    mutated_individ_only_u,
    select_by_tournament,
    separate_crossover_only_u,
    fitness_frob_norm_only_u,
    mutation_quadrant_gauss

)
from evo_storage import EvoStorage
from viz import components_comparison


def evolution_only_u_component(source_matrix, crossover, fitness, quadrant_idx):
    mutation = partial(mutation_quadrant_gauss, quadrant_idx=quadrant_idx,
                       mu=0, sigma=0.3, prob_global=0.1)
    init_population = partial(initial_pop_only_u_fixed_quadrant,
                              source_matrix=source_matrix, bound_value=0.5,
                              quadrant_idx=quadrant_idx)
    evo_operators = {'fitness': fitness,
                     'parent_selection': partial(select_by_tournament, tournament_size=20),
                     'mutation': partial(mutated_individ_only_u, mutate=mutation),
                     'crossover': partial(separate_crossover_only_u, crossover=crossover),
                     'initial_population': init_population}
    meta_params = {'pop_size': 200, 'generations': 3000, 'bound_value': 0.5,
                   'selection_rate': 0.2, 'crossover_rate': 0.60, 'random_selection_rate': 0.2, 'mutation_rate': 0.1}

    return evo_operators, meta_params


def run_evolution(source_matrix, crossover,
                  fitness, runs=10, quadrant_idx=1, **kwargs):
    if 'storage_path' in kwargs:
        storage = EvoStorage(dump_file_path=kwargs['storage_path'],
                             from_file=kwargs['storage_path'])
    else:
        storage = EvoStorage(dump_file_path='history.db')

    run_key = 'with_rot'
    if 'run_key' in kwargs:
        run_key = kwargs['run_key']

    evo_operators, meta_params = \
        evolution_only_u_component(source_matrix=source_matrix,
                                   crossover=crossover,
                                   fitness=fitness,
                                   quadrant_idx=quadrant_idx)
    evo_history = EvoHistory(description=run_key)

    for run_id in range(runs):
        print(f'run_id: {run_id}')
        evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params,
                                        history=evo_history, source_matrix=source_matrix)
        evo_strategy.run()
        best_solution = evo_strategy.graded_by_fitness()[0]
        worst_solution = evo_strategy.graded_by_fitness()[-1]

        u_best = best_solution.genotype[0]
        u_baseline, _, _ = np.linalg.svd(source_matrix)

        print(f'F-norm: {np.linalg.norm(u_best - u_baseline)}')
        print(f'Eigen values: {np.linalg.eig(u_baseline - u_best)[0]}')
        components_comparison([best_solution.genotype[0], worst_solution.genotype[0],
                               best_solution.genotype[0] - worst_solution.genotype[0],
                               best_solution.genotype[0] - u_baseline])
    evo_history.fitness_history_boxplots(values_to_plot='normed_min', save_to_file=False, gens_ticks=25)
    storage.save_run(key=run_key, evo_history=evo_history)


if __name__ == '__main__':
    source_matrix = np.random.rand(10, 10)
    quadrant_idx = 2
    crossover = partial(quadrant_increasing_dynamic_geo_crossover,
                        box_size_initial=1, quadrant_idx=quadrant_idx)
    eig_fitness = fitness_frob_norm_only_u
    run_evolution(source_matrix=source_matrix, crossover=crossover,
                  fitness=eig_fitness, runs=1, quadrant_idx=quadrant_idx)
