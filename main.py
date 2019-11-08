from evo_algorithm import BasicEvoStrategy

from evo_operators import (
    fitness_frob_norm,
    new_individ_random
)

if __name__ == '__main__':
    pop_size, generations = 10, 10
    evo_strategy = BasicEvoStrategy(new_individ=new_individ_random, fitness=fitness_frob_norm,
                                    pop_size=pop_size, generations=generations)

    evo_strategy.run()
