import numpy as np

from evo_algorithm import (
    BasicEvoStrategy,
    EvoHistory
)
from evo_operators import (
    fitness_frob_norm,
    new_individ_random_svd
)

if __name__ == '__main__':
    source_matrix = np.random.rand(10, 10)
    evo_operators = {'new_individ': new_individ_random_svd,
                     'fitness': fitness_frob_norm}
    meta_params = {'pop_size': 100,
                   'generations': 100}

    evo_history = EvoHistory()

    for run_id in range(5):
        print(f'run_id: {run_id}')
        evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params,
                                        history=evo_history, source_matrix=source_matrix)
        evo_strategy.run()

    evo_history.loss_history_boxplots(values_to_plot='min', save_to_file=False, gens_ticks=5)
