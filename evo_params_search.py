from functools import partial
from itertools import product

import numpy as np

from evo_algorithm import (
    BasicEvoStrategy,
    EvoHistory
)
from evo_operators import (
    select_by_tournament,
    mutation_gauss,
    k_point_crossover,
    initial_population_only_u_random,
    fitness_frob_norm_only_u,
    mutated_individ_only_u,
    separate_crossover_only_u
)


def default_only_u_configuration():
    source_matrix = np.random.rand(10, 10)
    mutation = partial(mutation_gauss, mu=0, sigma=0.25, prob_global=0.1)
    crossover = partial(k_point_crossover, type='random', k=4)
    init_population = partial(initial_population_only_u_random, source_matrix=source_matrix, bound_value=10.0)
    evo_operators = {'fitness': fitness_frob_norm_only_u,
                     'parent_selection': partial(select_by_tournament, tournament_size=20),
                     'mutation': partial(mutated_individ_only_u, mutate=mutation),
                     'crossover': partial(separate_crossover_only_u, crossover=crossover),
                     'initial_population': init_population}
    meta_params = {'pop_size': 500, 'generations': 1000, 'bound_value': 10.0,
                   'selection_rate': 0.2, 'crossover_rate': 0.6, 'random_selection_rate': 0.2, 'mutation_rate': 0.3}

    return source_matrix, evo_operators, meta_params


def run_evolution():
    source_matrix, evo_operators, meta_params = default_only_u_configuration()
    evo_history = EvoHistory()

    for run_id in range(1):
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
    evo_history.loss_history_boxplots(values_to_plot='min', save_to_file=False, gens_ticks=25)


def mutation_search():
    sigma_range = [0.3]
    prob_global = [0.05]
    mutation_rate = [0.2]

    source_matrix, evo_operators, default_meta_params = default_only_u_configuration()

    params_combinations = product(sigma_range, prob_global, mutation_rate)

    for params in params_combinations:
        sigma, prob_global, mutation_rate = params

        print(f'### PARAMS: sigma = {sigma}; prob_global = {prob_global}; mutation_rate = {mutation_rate}')

        mutation = partial(mutation_gauss, mu=0, sigma=sigma, prob_global=prob_global)
        evo_operators['mutation'] = partial(mutated_individ_only_u, mutate=mutation)
        meta_params = default_meta_params.copy()
        meta_params['mutation_rate'] = mutation_rate
        meta_params['pop_size'] = 200
        meta_params['generations'] = 500
        evo_history = EvoHistory()
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
        title = f'only_u_sigma={sigma:.2f};prob_global={prob_global:.2f};mutation_rate={mutation_rate:.2f}'
        evo_history.loss_history_boxplots(values_to_plot='min', save_to_file=True,
                                          dir='runs_history/18.11.19/mutation_search',
                                          title=title, gens_ticks=25)


if __name__ == '__main__':
    mutation_search()
