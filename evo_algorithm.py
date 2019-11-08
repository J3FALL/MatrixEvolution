import numpy as np

from evo_operators import (
    single_point_crossover,
    mutation_gauss
)


class BasicEvoStrategy:
    def __init__(self, evo_operators: dict, meta_params: dict, source_matrix):

        self.new_individ = evo_operators['new_individ']
        self.fitness = evo_operators['fitness']
        self.pop_size = meta_params['pop_size']
        self.generations = meta_params['generations']
        self.pop = []
        self.cur_gen = -1
        self.source_matrix = source_matrix
        self.matrix_size = source_matrix.shape

    def run(self):
        self.__init_population()
        while not self.__stop_criteria():
            print(self.cur_gen)
            self.__assign_fitness_values()
            top = self.__graded_by_fitness()[0]
            print(f'Best candidate with fitness: {top.fitness_value}')
            new_pop = self.__new_offspring()

            mutations_amount = int(len(new_pop) * 0.5)
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
        graded = self.__graded_by_fitness()
        k_best = int(len(graded) * 0.25)
        offspring = select_k_best(candidates=graded, k=k_best)
        childs_total = int(len(graded) * 0.75)
        childs_amount = 0
        while childs_amount < childs_total:
            parent_first, parent_second = np.random.choice(offspring), np.random.choice(offspring)

            # TODO: refactor this
            u_first, u_second = single_point_crossover(parent_first=parent_first.genotype[0],
                                                       parent_second=parent_second.genotype[0])
            s_first, s_second = single_point_crossover(parent_first=parent_first.genotype[1],
                                                       parent_second=parent_second.genotype[1])
            vh_first, vh_second = single_point_crossover(parent_first=parent_first.genotype[2],
                                                         parent_second=parent_second.genotype[2])

            child_first = MatrixIndivid(genotype=(u_first, s_first, vh_first))
            child_second = MatrixIndivid(genotype=(u_second, s_second, vh_second))

            # for val in ['u', 's', 'vh']:
            #     first_val, second_val = getattr(parent_first, val), getattr(parent_second, val)
            #
            #     single_point_crossover(parent_first=first_val, parent_second=second_val)

            offspring.extend([child_first, child_second])
            childs_amount += 2

        return offspring

    def __stop_criteria(self):
        return self.cur_gen >= self.generations


class MatrixIndivid:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness_value = None


def select_k_best(candidates, k):
    assert k <= len(candidates)
    return candidates[:k]


def mutated_individ(source_individ):
    u_mutated = mutation_gauss(candidate=source_individ.genotype[0], mu=0, sigma=0.2, prob_global=0.3)
    s_mutated = mutation_gauss(candidate=source_individ.genotype[1], mu=0, sigma=0.2, prob_global=0.3)
    vh_mutated = mutation_gauss(candidate=source_individ.genotype[2], mu=0, sigma=0.2, prob_global=0.3)

    resulted = MatrixIndivid(genotype=(u_mutated, s_mutated, vh_mutated))

    return resulted
