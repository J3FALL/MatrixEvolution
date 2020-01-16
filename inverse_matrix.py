from functools import partial

import numpy as np

from evo.operators.crossover import (
    increasing_dynamic_geo_crossover,
    crossover_inverse
)
from evo.operators.fitness import fitness_inverse_matrix_frob_norm
from evo.operators.init_population import initial_population_inverse_random
from evo.operators.mutation import (
    mutation_gauss,
    mutated_individ_inverse
)
from evo.operators.selection import select_by_tournament
from evo_algorithm import (
    BasicEvoStrategy,
    EvoHistory
)


def evo_configuration(source_matrix):
    inner_mutation = partial(mutation_gauss, mu=0, sigma=0.3, prob_global=0.05)
    inner_crossover = partial(increasing_dynamic_geo_crossover,
                              box_size_initial=1)

    fitness = fitness_inverse_matrix_frob_norm
    selection = partial(select_by_tournament, tournament_size=20)
    mutation = partial(mutated_individ_inverse, mutate=inner_mutation)
    crossover = partial(crossover_inverse, crossover=inner_crossover)
    init_population = partial(initial_population_inverse_random,
                              source_matrix=source_matrix, bound_value=1.0)

    evo_operators = {'fitness': fitness,
                     'parent_selection': selection,
                     'mutation': mutation,
                     'crossover': crossover,
                     'initial_population': init_population}
    meta_params = {'pop_size': 100, 'generations': 1000, 'bound_value': 1.0,
                   'selection_rate': 0.1, 'crossover_rate': 0.80,
                   'random_selection_rate': 0.1, 'mutation_rate': 0.3}

    return evo_operators, meta_params


if __name__ == '__main__':
    source_matrix = np.random.rand(10, 10)
    evo_operators, meta_params = evo_configuration(source_matrix=source_matrix)
    evo_history = EvoHistory(description=0)

    evo_strategy = BasicEvoStrategy(evo_operators=evo_operators,
                                    meta_params=meta_params,
                                    history=evo_history,
                                    source_matrix=source_matrix)
    evo_strategy.run()
