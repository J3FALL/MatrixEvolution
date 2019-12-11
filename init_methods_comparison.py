from functools import partial

import numpy as np

from evo_operators import (
    k_point_crossover,
    geo_crossover
)
from evo_storage import EvoStorage
from init_randomly import evo_random
from init_rotation import evo_with_rotations
from viz import joint_convergence_boxplots


def comparison_plot():
    storage = EvoStorage(dump_file_path='history.db', from_file='history.db')

    # TODO: keys of runs as external params
    random_run = storage.run_by_key(key='random_k_point')
    rot_run = storage.run_by_key(key='random_geo')

    joint_convergence_boxplots(history_runs=[random_run, rot_run], values_to_plot='min', gens_ticks=1)


def evo_random_vs_rotations():
    source_matrix = np.random.rand(10, 10)
    crossover = partial(k_point_crossover, type='random', k=4)
    runs = 10
    evo_random(source_matrix=source_matrix, runs=runs, crossover=crossover, run_key='random_k_point')
    evo_random(source_matrix=source_matrix, runs=runs, crossover=geo_crossover, run_key='random_geo',
               storage_path='history.db')
    comparison_plot()


def geo_vs_k_point_crossover():
    source_matrix = np.random.rand(10, 10)
    runs = 10
    evo_random(source_matrix=source_matrix, runs=runs, crossover=geo_crossover)
    evo_with_rotations(source_matrix=source_matrix, runs=runs, crossover=geo_crossover)

    storage = EvoStorage(dump_file_path='history.db', from_file='history.db')

    k_point_run = storage.run_by_key(key='random_geo')
    geo_run = storage.run_by_key(key='with_rot_geo')

    joint_convergence_boxplots(history_runs=[k_point_run, geo_run], values_to_plot='min', gens_ticks=1)


if __name__ == '__main__':
    evo_random_vs_rotations()
    # geo_vs_k_point_crossover()
