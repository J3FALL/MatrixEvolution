import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evo_algorithm import (
    EvoHistory
)


def joint_convergence_boxplots(history_runs: List[EvoHistory], values_to_plot='min', save_to_file=False,
                               dir='', title='Fitness history by generations', gens_ticks=5):
    lowest_value = 10e9
    for run in history_runs:
        reduced_values = run.reduced_history_values(values_to_plot=values_to_plot, gens_ticks=gens_ticks).tolist()
        lowest_value = min(lowest_value, np.min(reduced_values))

    all_runs = joint_dataframe(all_runs=history_runs, gens_ticks=gens_ticks)
    sns.boxplot(x='gen', y='fitness', hue='config', data=all_runs)

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


def joint_dataframe(all_runs: List[EvoHistory], gens_ticks):
    df = pd.DataFrame({'config': [], 'gen': [], 'run_id': [], 'fitness': []})
    for run in all_runs:
        reduced_values = run.reduced_history_values(gens_ticks=gens_ticks)
        gens_total, runs_total = len(reduced_values), len(reduced_values[0])
        for gen in range(gens_total):
            for run_id in range(runs_total):
                run_label = run.description
                df = df.append(
                    pd.Series(
                        {'config': run_label, 'gen': gen * gens_ticks, 'run_id': run_id,
                         'fitness': reduced_values[gen, run_id]}),
                    ignore_index=True
                )
    df.gen = df.gen.astype(int)

    return df


def components_comparison(values: List[np.ndarray]):
    values_amount = len(values)
    size = values[0].shape[0]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for idx in range(values_amount):
        fig_i, fig_j = divmod(idx, 2)
        value = values[idx]
        axs[fig_i, fig_j].imshow(value)
        for i in range(size):
            for j in range(size):
                axs[fig_i, fig_j].text(j, i, np.round(value[i, j], 2),
                                       ha="center", va="center", color="w")
        axs[fig_i, fig_j].set_xticklabels(np.arange(size))
        axs[fig_i, fig_j].set_yticklabels(np.arange(size))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
