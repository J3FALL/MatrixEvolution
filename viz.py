import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evo_algorithm import EvoHistory


# TODO: fix graphs, add test
def joint_convergence_boxplots(history_runs: List[EvoHistory], values_to_plot='min', save_to_file=False,
                               dir='', title='Fitness history by generations', gens_ticks=5):
    gens_total = history_runs[0].generations_amount()
    reduced_gens = [gen for gen in range(0, gens_total, gens_ticks)]

    lowest_value = 10e9
    for run in history_runs:
        reduced_values = run.reduced_history_values(values_to_plot=values_to_plot, gens_ticks=gens_ticks).tolist()
        sns.boxplot(reduced_gens, reduced_values)

        plt.plot([], [], ' ', label=f'Run # {history_runs.index(run)}')

        lowest_value = min(lowest_value, np.min(reduced_values))

    plt.title(title)
    plt.ylabel('Fitness')
    plt.xlabel('Generation, #')

    plt.plot([], [], ' ', label=f'Lowest value in history: {lowest_value}')
    plt.legend()

    if save_to_file:
        plt.savefig(os.path.join(dir, f'{title}.png'))
        plt.close()
    else:
        plt.show()
