import numpy as np

from evo_storage import EvoStorage
from init_randomly import evo_random
from init_rotation import evo_with_rotations
from viz import joint_convergence_boxplots


def comparison_plot():
    storage = EvoStorage(dump_file_path='history.db', from_file='history.db')

    random_run = storage.run_by_key(key='random')
    rot_run = storage.run_by_key(key='with_rot')

    joint_convergence_boxplots(history_runs=[random_run, rot_run], values_to_plot='min', gens_ticks=5)


def evo_random_vs_rotations():
    source_matrix = np.random.rand(20, 20)

    evo_random(source_matrix=source_matrix)
    evo_with_rotations(source_matrix=source_matrix)

    comparison_plot()


if __name__ == '__main__':
    evo_random_vs_rotations()
