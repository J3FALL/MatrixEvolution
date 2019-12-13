from functools import partial

import numpy as np

from evo_operators import (
    k_point_crossover,
    geo_crossover_fixed_box
)
from evo_storage import EvoStorage
from init_randomly import evo_random
from viz import joint_convergence_boxplots


def comparison_plot(run_key_first, run_key_second):
    storage = EvoStorage(dump_file_path='history.db', from_file='history.db')

    random_run = storage.run_by_key(key=run_key_first)
    rot_run = storage.run_by_key(key=run_key_second)

    joint_convergence_boxplots(history_runs=[random_run, rot_run], values_to_plot='min', gens_ticks=10)


def evo_random_vs_rotations():
    source_matrix = np.random.rand(10, 10)
    k_point = partial(k_point_crossover, type='random', k=4)
    runs = 20
    evo_random(source_matrix=source_matrix, runs=runs, crossover=k_point,
               run_key='random_k_point')
    evo_random(source_matrix=source_matrix, runs=runs, crossover=partial(geo_crossover_fixed_box, box_size=5),
               run_key='random_box_5',
               storage_path='history.db')
    # evo_random(source_matrix=source_matrix, runs=runs, crossover=crossover, run_key='random_k_point')
    # evo_random(source_matrix=source_matrix, runs=runs, crossover=geo_crossover, run_key='random_geo',
    #            storage_path='history.db')
    comparison_plot(run_key_first='random_k_point', run_key_second='random_box_5')


if __name__ == '__main__':
    evo_random_vs_rotations()
