import numpy as np

from evo_algorithm import BasicEvoStrategy
from evo_operators import (
    fitness_frob_norm,
    new_individ_random_svd
)

if __name__ == '__main__':
    source_matrix = np.random.rand(100, 100)
    evo_operators = {'new_individ': new_individ_random_svd,
                     'fitness': fitness_frob_norm}
    meta_params = {'pop_size': 10,
                   'generations': 10}
    evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params, source_matrix=source_matrix)

    evo_strategy.run()
