from functools import partial

import numpy as np

from evo_operators import (
    k_point_crossover,
    dynamic_geo_crossover,
    geo_crossover_fixed_box
)
from evo_storage import EvoStorage
from init_randomly import evo_random
from viz import joint_convergence_boxplots


def comparison_plot(run_key_first, run_key_second):
    storage = EvoStorage(dump_file_path='history.db', from_file='history.db')

    run_first = storage.run_by_key(key=run_key_first)
    run_second = storage.run_by_key(key=run_key_second)

    joint_convergence_boxplots(history_runs=[run_first, run_second], values_to_plot='min', gens_ticks=50)


def evo_random_vs_rotations():
    source_matrix = np.random.rand(20, 20)
    k_point = partial(k_point_crossover, type='random', k=4)
    geo_fixed = partial(geo_crossover_fixed_box, box_size=3)
    runs = 10
    evo_random(source_matrix=source_matrix, runs=runs, crossover=k_point,
               run_key='random_k_point')
    dynamic_crossover = partial(dynamic_geo_crossover, box_size_initial=10)
    evo_random(source_matrix=source_matrix, runs=runs, crossover=dynamic_crossover,
               run_key='random_dynamic_box',
               storage_path='history.db')
    # evo_random(source_matrix=source_matrix, runs=runs, crossover=crossover, run_key='random_k_point')
    # evo_random(source_matrix=source_matrix, runs=runs, crossover=geo_crossover, run_key='random_geo',
    #            storage_path='history.db')
    comparison_plot(run_key_first='random_k_point', run_key_second='random_dynamic_box')


if __name__ == '__main__':
    evo_random_vs_rotations()
