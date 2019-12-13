import matplotlib.pyplot as plt
import numpy as np


class MatrixIndivid:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness_value = None

    def plot_value(self, component_idx=0):
        value = self.genotype[component_idx]
        size = value.shape[0]

        plt.imshow(value)

        for i in range(size):
            for j in range(size):
                plt.text(j, i, np.round(value[i, j], 3),
                         ha="center", va="center", color="w")

        plt.xticks(np.arange(size))
        plt.yticks(np.arange(size))
        plt.show()


if __name__ == '__main__':
    source_matrix = np.random.rand(10, 10)
    u, s, vh = np.linalg.svd(source_matrix, full_matrices=True)
    individ = MatrixIndivid(genotype=(u, s, vh))

    individ.plot_value(component_idx=0)
