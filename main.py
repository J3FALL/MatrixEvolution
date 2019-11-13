import numpy as np

from evo_algorithm import (
    BasicEvoStrategy,
    EvoHistory
)
from evo_operators import (
    new_individ_random_s_only,
    fitness_s_component_diff
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
    evo_operators = {'new_individ': new_individ_random_s_only,
                     'fitness': fitness_s_component_diff}
    meta_params = {'pop_size': 500,
                   'generations': 500}

    evo_history = EvoHistory()

    for run_id in range(5):
        print(f'run_id: {run_id}')
        evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params,
                                        history=evo_history, source_matrix=source_matrix)
        evo_strategy.run()
        best_solution = evo_strategy.graded_by_fitness()[0]
        singular_values = sorted(best_solution.genotype[1], reverse=True)
        print(singular_values)
        _, s, _ = np.linalg.svd(source_matrix)
        print(s)
        print(np.abs(singular_values - s))
    evo_history.loss_history_boxplots(values_to_plot='min', save_to_file=False, gens_ticks=25)
