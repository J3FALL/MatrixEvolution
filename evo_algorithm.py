import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evo_operators import (
    mutation_gauss,
    single_point_crossover,
    select_by_tournament
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

            top = self.__graded_by_fitness()[0]
            print(f'Best candidate with fitness: {top.fitness_value}')
            new_pop = self.__new_offspring()

            mutations_amount = int(len(new_pop) * 0.1)
            for _ in range(mutations_amount):
                idx = np.random.randint(len(new_pop) - 1)
                new_pop[idx] = mutated_individ(new_pop[idx])
            self.pop = new_pop
            self.cur_gen += 1

    def __init_population(self):
        for _ in range(self.pop_size):
            individ = MatrixIndivid(genotype=self.new_individ(self.matrix_size))
            self.pop.append(individ)
        self.cur_gen = 0

    def __assign_fitness_values(self):
        for individ in self.pop:
            individ.fitness_value = self.fitness(source_matrix=self.source_matrix, svd=individ.genotype)

    def __graded_by_fitness(self):
        pop = np.copy(self.pop)
        return sorted(pop, key=lambda p: p.fitness_value)

    def __new_offspring(self):
        selected_amount = int(len(self.pop) * 0.1)
        offspring = select_by_tournament(candidates=self.pop, k=selected_amount, tournament_size=10)

        # Add some diversity
        random_chosen = np.random.choice(self.pop, int(len(self.pop) * 0.1))
        offspring.extend(random_chosen)

        childs_total = int(len(self.pop) * 0.8)
        childs_amount = 0

        while childs_amount < childs_total:
            parent_first, parent_second = np.random.choice(offspring), np.random.choice(offspring)
            child_first, child_second = separate_crossover(parent_first, parent_second)
            # for val in ['u', 's', 'vh']:
            #     first_val, second_val = getattr(parent_first, val), getattr(parent_second, val)
            #
            #     single_point_crossover(parent_first=first_val, parent_second=second_val)

            offspring.extend([child_first, child_second])
            childs_amount += 2

        return offspring

    def __stop_criteria(self):
        return self.cur_gen >= self.generations

    # TODO: implement callbacks logic
    def __history_callback(self):
        fitness = [ind.fitness_value for ind in self.pop]
        self.history.new_generation(avg_fitness=np.average(fitness), min_fitness_in_pop=np.min(fitness))


class MatrixIndivid:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness_value = None


def mutated_individ(source_individ):
    u_mutated = mutation_gauss(candidate=source_individ.genotype[0], mu=0, sigma=0.05, prob_global=0.05)
    s_mutated = mutation_gauss(candidate=source_individ.genotype[1], mu=0, sigma=0.05, prob_global=0.05)
    vh_mutated = mutation_gauss(candidate=source_individ.genotype[2], mu=0, sigma=0.05, prob_global=0.05)

    resulted = MatrixIndivid(genotype=(u_mutated, s_mutated, vh_mutated))

    return resulted


class EvoHistory:
    def __init__(self):
        self.__history = {}
        self.last_run_idx = -1

    def init_new_run(self):
        self.last_run_idx += 1
        self.__history[self.last_run_idx] = []

    def new_generation(self, avg_fitness, min_fitness_in_pop):
        if self.last_run_idx < 0:
            self.init_new_run()
        self.__history[self.last_run_idx].append(
            [avg_fitness, min_fitness_in_pop]
        )

    # TODO: refactor this
    def loss_history_boxplots(self, values_to_plot='min', save_to_file=False,
                              dir='', title='Fitness history by generations', gens_ticks=5):
        gens = [gen for gen in range(len(self.__history[0]))]

        avg_fitness_by_gens = []

        value_idx = 1 if values_to_plot == 'min' else 0

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

    u_first, u_second = single_point_crossover(parent_first=parent_first.genotype[0],
                                               parent_second=parent_second.genotype[0],
                                               type=crossover_type)
    s_first, s_second = single_point_crossover(parent_first=parent_first.genotype[1],
                                               parent_second=parent_second.genotype[1],
                                               type='horizontal')
    vh_first, vh_second = single_point_crossover(parent_first=parent_first.genotype[2],
                                                 parent_second=parent_second.genotype[2],
                                                 type=crossover_type)

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
