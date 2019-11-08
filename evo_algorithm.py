import numpy as np


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
            graded = self.__graded_by_fitness()

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

    def __stop_criteria(self):
        return self.cur_gen >= self.generations


class MatrixIndivid:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness_value = None
