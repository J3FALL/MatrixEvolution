from functools import partial

import numpy as np

from evo.operators.crossover import (
    k_point_crossover,
    increasing_dynamic_geo_crossover
)
from evo.operators.fitness import (
    fitness_svd_frob_norm_only_u,
    fitness_svd_combined_norm_only_u)
from evo_storage import EvoStorage
from init_randomly import evo_random
from viz import joint_convergence_boxplots


def comparison_plot(run_key_first, run_key_second):
    storage = EvoStorage(dump_file_path='history.db', from_file='history.db')

    run_first = storage.run_by_key(key=run_key_first)
    run_second = storage.run_by_key(key=run_key_second)

    joint_convergence_boxplots(history_runs=[run_first, run_second], values_to_plot='normed_min', gens_ticks=25)


def compare_methods():
    source_matrix = np.random.rand(10, 10)
    k_point = partial(k_point_crossover, type='random', k=4)
    runs = 10
    inc_dynamic_geo_crossover = partial(increasing_dynamic_geo_crossover, box_size_initial=1)
    evo_random(source_matrix=source_matrix, runs=runs, crossover=inc_dynamic_geo_crossover,
               fitness=fitness_svd_frob_norm_only_u, run_key='dynamic_normed_frob')
    evo_random(source_matrix=source_matrix, runs=runs, crossover=inc_dynamic_geo_crossover,
               fitness=fitness_svd_combined_norm_only_u, run_key='dynamic_normed_2+frob',
               storage_path='history.db')

    comparison_plot(run_key_first='dynamic_normed_frob', run_key_second='dynamic_normed_2+frob')


if __name__ == '__main__':
    compare_methods()
