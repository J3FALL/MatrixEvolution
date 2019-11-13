import os
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pysampling.sample import sample

from evo_operators import (
    mutation_gauss,
    single_point_crossover,
    select_by_tournament,
    select_k_best,
    k_point_crossover
)


class BasicEvoStrategy:
    def __init__(self, evo_operators: dict, meta_params: dict, history, source_matrix):

        self.new_individ = evo_operators['new_individ']
        self.fitness = evo_operators['fitness']
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

            mutations_amount = int(len(offspring) * 0.2)
            for _ in range(mutations_amount):
                idx = np.random.randint(len(offspring) - 1)
                offspring[idx] = mutated_individ_only_s(offspring[idx])
            self.pop = offspring
            self.cur_gen += 1

    def __init_population(self):
        self.pop = initial_population_from_lhs_only_s(samples_amount=self.pop_size, vector_size=self.matrix_size[0],
                                                      values_range=10.0, source_matrix=self.source_matrix)
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
        selected_amount = int(len(self.pop) * 0.2)
        selected_parents = select_by_tournament(candidates=self.pop, k=selected_amount, tournament_size=20)
        offspring.extend(selected_parents)

        childs_total = int(len(self.pop) * 0.7)
        childs_amount = 0

        while childs_amount < childs_total:
            parent_first, parent_second = np.random.choice(selected_parents), np.random.choice(selected_parents)
            child_first, child_second = separate_crossover_only_s(parent_first, parent_second)
            # for val in ['u', 's', 'vh']:
            #     first_val, second_val = getattr(parent_first, val), getattr(parent_second, val)
            #
            #     single_point_crossover(parent_first=first_val, parent_second=second_val)

            offspring.extend([child_first, child_second])
            childs_amount += 2
        random_chosen = self.__diversity(rate=0.1, fraction_worst=0.5)
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


def initial_population_from_lhs_only_s(samples_amount, vector_size, values_range, source_matrix):
    print('Sampling from LHS...')
    s_samples = values_range * sample("lhs", samples_amount, vector_size) - values_range / 2.0
    print('Sampling: done')
    u_base, _, vh_base = np.linalg.svd(source_matrix, full_matrices=True)

    pop = []
    for idx, s in enumerate(s_samples):
        pop.append(MatrixIndivid(genotype=(u_base, s, vh_base)))
    return pop


class MatrixIndivid:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness_value = None


def mutated_individ(source_individ):
    u_mutated = mutation_gauss(candidate=source_individ.genotype[0], mu=0, sigma=0.2, prob_global=0.05)
    s_mutated = mutation_gauss(candidate=source_individ.genotype[1], mu=0, sigma=0.2, prob_global=0.05)
    vh_mutated = mutation_gauss(candidate=source_individ.genotype[2], mu=0, sigma=0.2, prob_global=0.05)

    resulted = MatrixIndivid(genotype=(u_mutated, s_mutated, vh_mutated))

    return resulted


def mutated_individ_only_s(source_individ):
    u, s, vh = source_individ.genotype
    u_resulted = np.copy(u)
    vh_resulted = np.copy(vh)

    s_mutated = mutation_gauss(candidate=s, mu=0, sigma=1.0, prob_global=0.25)

    resulted = MatrixIndivid(genotype=(u_resulted, s_mutated, vh_resulted))

    return resulted


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


def separate_crossover(parent_first, parent_second):
    crossover_type = np.random.choice(['horizontal', 'vertical'])
    u_first, u_second = k_point_crossover(parent_first=parent_first.genotype[0],
                                          parent_second=parent_second.genotype[0],
                                          type=crossover_type, k=4)
    s_first, s_second = k_point_crossover(parent_first=parent_first.genotype[1],
                                          parent_second=parent_second.genotype[1],
                                          type='horizontal', k=4)
    vh_first, vh_second = k_point_crossover(parent_first=parent_first.genotype[2],
                                            parent_second=parent_second.genotype[2],
                                            type=crossover_type, k=4)

    child_first = MatrixIndivid(genotype=(u_first, s_first, vh_first))
    child_second = MatrixIndivid(genotype=(u_second, s_second, vh_second))

    return child_first, child_second


def separate_crossover_only_s(parent_first, parent_second):
    u_first, u_second = parent_first.genotype[0], parent_second.genotype[0],

    s_first, s_second = k_point_crossover(parent_first=parent_first.genotype[1],
                                          parent_second=parent_second.genotype[1],
                                          type='horizontal', k=4)
    vh_first, vh_second = parent_first.genotype[2], parent_second.genotype[2]

    child_first = MatrixIndivid(genotype=(u_first, s_first, vh_first))
    child_second = MatrixIndivid(genotype=(u_second, s_second, vh_second))

    return child_first, child_second


def joint_crossover(parent_first, parent_second):
    # TODO: refactor this
    crossover_type = np.random.choice(['horizontal', 'vertical'])

    min_size = np.min(parent_first.genotype[0].shape)
    cross_point = np.random.randint(0, min_size - 1)
    u_first, u_second = single_point_crossover(parent_first=parent_first.genotype[0],
                                               parent_second=parent_second.genotype[0],
                                               type=crossover_type, cross_point=cross_point)
    s_first, s_second = single_point_crossover(parent_first=parent_first.genotype[1],
                                               parent_second=parent_second.genotype[1],
                                               type='horizontal', cross_point=cross_point)
    vh_first, vh_second = single_point_crossover(parent_first=parent_first.genotype[2],
                                                 parent_second=parent_second.genotype[2],
                                                 type=crossover_type, cross_point=cross_point)

    child_first = MatrixIndivid(genotype=(u_first, s_first, vh_first))
    child_second = MatrixIndivid(genotype=(u_second, s_second, vh_second))

    return child_first, child_second


def svd_frob_norm(best_candidate, matrix):
    u_base, s_base, vh_base = np.linalg.svd(matrix, full_matrices=True)

    u, s, vh = best_candidate

    return np.linalg.norm(u - u_base), np.linalg.norm(s - s_base), np.linalg.norm(vh - vh_base)
