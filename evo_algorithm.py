class BasicEvoStrategy:
    def __init__(self, new_individ, fitness, pop_size, generations):
        self.new_individ = new_individ
        self.fitness = fitness
        self.pop_size = pop_size
        self.generations = generations
        self.pop = []
        self.cur_gen = -1
        self.matrix_size = (100, 100)

    def run(self):
        self.__init_population()
        while not self.__stop_criteria():
            print(self.cur_gen)

            self.cur_gen += 1

    def __init_population(self):
        self.pop = [self.new_individ(self.matrix_size) for _ in range(self.pop_size)]
        self.cur_gen = 0

    def __stop_criteria(self):
        return self.cur_gen >= self.generations
