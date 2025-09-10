import random
import numpy as np


class Individual:
    def __init__(self, n, chromosome=None):
        if chromosome is None:
            self.chromosome = np.random.randint(0, 2, size=n)
        else:
            self.chromosome = chromosome
        self.fitness = None

    def evaluate(self, problem):
        self.fitness = problem(self.chromosome)
        return self.fitness
    
    # Flip a single random bit
    def singleRandomMutation(self):
        # Select a random index in the chromosome
        idx = random.randint(0, len(self.chromosome) - 1)
        # Flip the bit at the selected index
        self.chromosome[idx] = 1 - self.chromosome[idx]
        return self
    
    # Flip all bits with prob 1/n
    def flipAllBitsMutation(self):
        n = len(self.chromosome)
        for i in range(n):
            if np.random.rand() < 1.0 / n:
                #Flip the bit
                self.chromosome[i] = 1 - self.chromosome[i]