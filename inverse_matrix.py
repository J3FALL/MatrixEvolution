from functools import partial

import numpy as np

from evo.operators.crossover import (
    crossover_inverse,
    swap_crossover,
    increasing_dynamic_geo_crossover,
    arithmetic_swap_crossover
)
from evo.operators.fitness import fitness_inverse_matrix_frob_norm
from evo.operators.init_population import (
    initial_population_inverse_random,
    random_matrix,
    initial_population_inverse_external)
from evo.operators.mutation import (
    mutation_gauss,
    mutated_individ_inverse
)
from evo.operators.selection import select_by_tournament
from evo_algorithm import (
    EvoHistory, BasicEvoStrategy
)
from viz import joint_convergence_boxplots


def random_population(pop_size, shape, bound_value=10.0):
    pop = []
    for _ in range(pop_size):
        pop.append(random_matrix(shape, bound_value=bound_value))
    return pop


def evo_configuration(source_matrix):
    inner_mutation = partial(mutation_gauss, mu=0, sigma=0.3, prob_global=0.10)
    inner_crossover = partial(swap_crossover, rows_to_swap=2)
    fitness = fitness_inverse_matrix_frob_norm
    selection = partial(select_by_tournament, tournament_size=20)
    mutation = partial(mutated_individ_inverse, mutate=inner_mutation)
    crossover = partial(crossover_inverse, crossover=inner_crossover)

    external_pop = random_population(pop_size=100, shape=source_matrix.shape,
                                     bound_value=1.0)

    init_population = partial(initial_population_inverse_random,
                              source_matrix=source_matrix, bound_value=1.0)
    init_population = partial(initial_population_inverse_external,
                              external_pop=external_pop)

    evo_operators = {'fitness': fitness,
                     'parent_selection': selection,
                     'mutation': mutation,
                     'crossover': crossover,
                     'initial_population': init_population}
    meta_params = {'pop_size': 100, 'generations': 10000, 'bound_value': 1.0,
                   'selection_rate': 0.3, 'crossover_rate': 0.70,
                   'random_selection_rate': 0.0, 'mutation_rate': 0.2}

    return evo_operators, meta_params


def _evo_config_with_dynamic_crossover(source_matrix):
    operators, meta_params = evo_configuration(source_matrix=source_matrix)
    dynamic_inner_crossover = partial(increasing_dynamic_geo_crossover,
                                      box_size_initial=1)
    crossover = partial(crossover_inverse, crossover=dynamic_inner_crossover)
    operators['crossover'] = crossover

    return operators, meta_params


def _evo_config_with_arithmetic_swap_crossover(source_matrix):
    operators, meta_params = evo_configuration(source_matrix=source_matrix)
    arithmetic_swap = partial(arithmetic_swap_crossover,
                              rows_to_swap=2, fraction=0.3)
    crossover = partial(crossover_inverse, crossover=arithmetic_swap)
    operators['crossover'] = crossover

    return operators, meta_params


def run_evolution(operators, meta_params, source_matrix, runs=10,
                  description='evo_history'):
    evo_history = EvoHistory(description=description)
    for run_id in range(runs):
        evo_strategy = BasicEvoStrategy(evo_operators=operators,
                                        meta_params=meta_params,
                                        history=evo_history,
                                        source_matrix=source_matrix)
        evo_strategy.run()

    return evo_history


def compare_configurations():
    source_matrix = np.random.rand(10, 10)
    swap_operators, meta_params = evo_configuration(source_matrix=source_matrix)

    # dynamic_operators, _ = _evo_config_with_dynamic_crossover(source_matrix)
    arith_swap_operators, _ = _evo_config_with_arithmetic_swap_crossover(source_matrix)
    runs = 2
    history_swap = run_evolution(operators=swap_operators,
                                 meta_params=meta_params,
                                 source_matrix=source_matrix,
                                 runs=runs, description='swap')

    history_dynamic = run_evolution(operators=arith_swap_operators,
                                    meta_params=meta_params,
                                    source_matrix=source_matrix,
                                    runs=runs, description='arith_swap')

    joint_convergence_boxplots(history_runs=[history_swap, history_dynamic],
                               values_to_plot='min', gens_ticks=100)


if __name__ == '__main__':
    compare_configurations()
