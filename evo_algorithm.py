import os
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evo_operators import (
    select_k_best,
    initial_population_from_lhs_only_s,
)
from matrix import (
    MatrixIndivid
)


class BasicEvoStrategy:
    def __init__(self, evo_operators: dict, meta_params: dict, history, source_matrix):
        self.new_individ = evo_operators['new_individ']
        self.fitness = evo_operators['fitness']
        self.select_parents = evo_operators['parent_selection']
        self.mutate = evo_operators['mutation']
        self.crossover = evo_operators['crossover']

        self.meta_params = meta_params
        self.pop_size = meta_params['pop_size']
        self.generations = meta_params['generations']
        self.pop = []
        self.cur_gen = -1
        self.source_matrix = source_matrix
        self.matrix_size = source_matrix.shape

        self.history = history

    def run(self):
        self.history.init_new_run()
        self.__init_population()
        while not self.__stop_criteria():
            print(self.cur_gen)
            self.__assign_fitness_values()
            self.__history_callback()

            top = self.graded_by_fitness()[0]
            avg = np.average([individ.fitness_value for individ in self.pop])
            print(f'Best candidate with fitness: {top.fitness_value}')
            print(f'Average fitness in population: {avg}')

            offspring = self.__new_offspring()

            mutations_amount = int(len(offspring) * self.meta_params['mutation_rate'])
            for _ in range(mutations_amount):
                idx = np.random.randint(len(offspring) - 1)
                offspring[idx] = self.mutate(offspring[idx])
            self.pop = offspring
            self.cur_gen += 1

    def __init_population(self):
        self.pop = initial_population_from_lhs_only_s(samples_amount=self.pop_size, vector_size=self.matrix_size[0],
                                                      values_range=15.0, source_matrix=self.source_matrix)
        self.cur_gen = 0

    def __init_basically(self):
        # TODO: init_population as external function
        for _ in range(self.pop_size):
            individ = MatrixIndivid(genotype=self.new_individ(self.source_matrix))
            self.pop.append(individ)

    def __assign_fitness_values(self):
        for individ in self.pop:
            individ.fitness_value = self.fitness(source_matrix=self.source_matrix, svd=individ.genotype)

    def graded_by_fitness(self):
        self.__assign_fitness_values()
        graded = sorted(self.pop, key=attrgetter('fitness_value'))
        return graded

    def __new_offspring(self):
        offspring = []
        selected_amount = int(len(self.pop) * self.meta_params['selection_rate'])
        selected_parents = self.select_parents(candidates=self.pop, k=selected_amount)
        offspring.extend(selected_parents)

        childs_total = int(len(self.pop) * self.meta_params['crossover_rate'])
        childs_amount = 0

        while childs_amount < childs_total:
            parent_first, parent_second = np.random.choice(selected_parents), np.random.choice(selected_parents)
            child_first, child_second = self.crossover(parent_first, parent_second)
            # for val in ['u', 's', 'vh']:
            #     first_val, second_val = getattr(parent_first, val), getattr(parent_second, val)
            #
            #     single_point_crossover(parent_first=first_val, parent_second=second_val)

            offspring.extend([child_first, child_second])
            childs_amount += 2
        random_chosen = self.__diversity(rate=self.meta_params['random_selection_rate'], fraction_worst=0.5)
        offspring.extend(random_chosen)
        return offspring

    def __survived(self, survive_rate=0.1):
        survived = select_k_best(candidates=self.pop, k=int(len(self.pop) * survive_rate))
        return survived

    def __diversity(self, rate=0.1, fraction_worst=0.5):
        k_worst = int((1.0 - fraction_worst) * len(self.pop))
        worst_candidates = self.graded_by_fitness()[k_worst:]
        random_chosen = np.random.choice(worst_candidates, int(len(self.pop) * rate))

        return random_chosen

    def __stop_criteria(self):
        return self.cur_gen >= self.generations

    # TODO: implement callbacks logic
    def __history_callback(self):
        fitness = [ind.fitness_value for ind in self.pop]
        best_candidate = self.graded_by_fitness()[0]
        u_norm, s_norm, vh_norm = svd_frob_norm(best_candidate=best_candidate.genotype, matrix=self.source_matrix)
        _, s, _ = np.linalg.svd(self.source_matrix, full_matrices=True)

        self.history.new_generation(avg_fitness=np.average(fitness), min_fitness_in_pop=np.min(fitness),
                                    u_norm=u_norm, s_norm=s_norm, vh_norm=vh_norm)


class EvoHistory:
    def __init__(self):
        self.__history = {}
        self.last_run_idx = -1

    def init_new_run(self):
        self.last_run_idx += 1
        self.__history[self.last_run_idx] = []

    def new_generation(self, avg_fitness, min_fitness_in_pop, u_norm, s_norm, vh_norm):
        if self.last_run_idx < 0:
            self.init_new_run()
        self.__history[self.last_run_idx].append(
            [avg_fitness, min_fitness_in_pop, u_norm, s_norm, vh_norm]
        )

    # TODO: refactor this
    def loss_history_boxplots(self, values_to_plot='min', save_to_file=False,
                              dir='', title='Fitness history by generations', gens_ticks=5):
        gens = [gen for gen in range(len(self.__history[0]))]

        avg_fitness_by_gens = []

        values_by_idx = {
            'avg': 0,
            'min': 1,
            'u_norm': 2,
            's_norm': 3,
            'vh_norm': 4
        }
        value_idx = values_by_idx[values_to_plot]

        for gen in gens:
            avg_loss_by_gen = []
            for run_id in self.__history.keys():
                avg_loss = self.__history[run_id][gen][value_idx]
                avg_loss_by_gen.append(avg_loss)
            avg_fitness_by_gens.append(avg_loss_by_gen)

        reduced_gens = []
        reduced_loss = []

        for gen in gens:
            if gen % gens_ticks == 0:
                reduced_gens.append(gen)
                reduced_loss.append(avg_fitness_by_gens[gen])

        sns.boxplot(reduced_gens, reduced_loss, color="seagreen")

        plt.title(title)
        plt.ylabel('Fitness')
        plt.xlabel('Generation, #')

        if save_to_file:
            plt.savefig(os.path.join(dir, 'loss_history_boxplots.png'))
        else:
            plt.show()


def svd_frob_norm(best_candidate, matrix):
    u_base, s_base, vh_base = np.linalg.svd(matrix, full_matrices=True)

    u, s, vh = best_candidate

    return np.linalg.norm(u - u_base), np.linalg.norm(s - s_base), np.linalg.norm(vh - vh_base)
