from individual import Individual
import random
import math
import numpy as np
import copy


class Population:
    individuals: list[Individual]
    # Initialise a list of individuals as a population
    def __init__(self, size, n, problem):
        self.individuals = [Individual(n) for _ in range(size)]
        for ind in self.individuals:
            ind.evaluate(problem)

    # Return the best individual within the population
    def getBest(self) -> Individual | None:
        if not self.individuals:
            return None

        return max(self.individuals, key=lambda ind: ind.fitness)
        
    # Generalised function to centralise the execution of a crossover
    def performCrossover(self, parent1, parent2, crossover_probability=0.8) -> tuple[Individual, Individual]:
        # Should crossover occur, if not return parents without modification
        random_number: float = random.random()
        if random_number > crossover_probability:
            return Individual(len(parent1.chromosome), parent1.chromosome.copy()), \
            Individual(len(parent2.chromosome), parent2.chromosome.copy())

        # If crossover should occur, perform crossover and return children
        child1: Individual | None = None
        child2: Individual | None = None
        
        child1, child2 = self.uniform_Crossover(parent1, parent2)
        
        # if either child is None then the crossover was incomplete, return parents
        if child1 is None or child2 is None:
            return parent1, parent2

        return child1, child2

    # tournament selection for parent selection
    def tournament_Selection(self, k: int = 3) -> Individual:
        # Randomly select k individuals from the population
        tournament: list[Individual] = random.sample(self.individuals, k)
        # Get the individual with the lowest cost
        winner: Individual = max(tournament, key=lambda ind: ind.fitness)
        return winner

    # Informal tournament selection for parent selection
    # This method selects two parents using tournament selection
    # Not sure if right
    def informal_tournament_selection(self, k: int = 3) -> tuple[Individual,Individual]:
        parent1: Individual = self.tournament_Selection(k)
        parent2: Individual = self.tournament_Selection(k)
        return parent1, parent2
    
    
    def uniform_Crossover(self, parent1: Individual, parent2) -> tuple[Individual, Individual]:
        length: int = len(parent1.chromosome)
        mask: list[int] = list(np.random.randint(0, 2, size=length))
        child1: Individual = copy.deepcopy(parent1)
        child2: Individual = copy.deepcopy(parent2)
        for i in range(length):
            if mask[i] == 1:
                child1.chromosome[i] = parent2.chromosome[i]
                child2.chromosome[i] = parent1.chromosome[i]
        return child1, child2
        


