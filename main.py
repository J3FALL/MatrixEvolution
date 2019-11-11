import numpy as np

from evo_algorithm import (
    BasicEvoStrategy,
    EvoHistory
)
from evo_operators import (
    fitness_frob_norm,
    new_individ_random_svd
)


def compare_results(matrix, evo_results):
    u_base, s_base, vh_base = np.linalg.svd(matrix, full_matrices=True)

    u, s, vh = evo_results

    print(f'Frob-norm U: {np.linalg.norm(u - u_base)}')
    print(f'Frob-norm S: {np.linalg.norm(s - s_base)}')
    print(f'Frob-norm VH: {np.linalg.norm(vh - vh_base)}')

    print('###### U - U_base #####')
    print(np.sqrt(np.power(u - u_base, 2)))

    print('###### S - S_base #####')
    print(np.sqrt(np.power(s - s_base, 2)))

    print('###### Vh - Vh_base #####')
    print(np.sqrt(np.power(vh - vh_base, 2)))


if __name__ == '__main__':
    source_matrix = np.random.rand(10, 10)
    evo_operators = {'new_individ': new_individ_random_svd,
                     'fitness': fitness_frob_norm}
    meta_params = {'pop_size': 200,
                   'generations': 200}

    evo_history = EvoHistory()

    for run_id in range(5):
        print(f'run_id: {run_id}')
        evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params,
                                        history=evo_history, source_matrix=source_matrix)
        evo_strategy.run()
    # best_solution = evo_strategy.graded_by_fitness()[0]
    # compare_results(matrix=source_matrix, evo_results=best_solution.genotype)

    evo_history.loss_history_boxplots(values_to_plot='vh_norm', save_to_file=False, gens_ticks=10)
