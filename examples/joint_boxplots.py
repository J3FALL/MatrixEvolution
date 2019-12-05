from evo_algorithm import (
    EvoHistory,
    BasicEvoStrategy
)
from init_randomly import evolution_only_u_component
from viz import joint_convergence_boxplots

if __name__ == '__main__':
    source_matrix, evo_operators, meta_params = evolution_only_u_component()
    evo_history_first = EvoHistory(description='First')

    for run_id in range(5):
        print(f'run_id: {run_id}')
        evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params,
                                        history=evo_history_first, source_matrix=source_matrix)
        evo_strategy.run()

    evo_history_second = EvoHistory(description='Second')

    for run_id in range(5):
        print(f'run_id: {run_id}')
        evo_strategy = BasicEvoStrategy(evo_operators=evo_operators, meta_params=meta_params,
                                        history=evo_history_second, source_matrix=source_matrix)
        evo_strategy.run()

    joint_convergence_boxplots(history_runs=[evo_history_first, evo_history_second], gens_ticks=1)
