from ioh import get_problem, ProblemType, ProblemClass, logger
import random
import numpy as np
import time


class Ant:
    def __init__(self, problem: ProblemType):
        self.problem = problem
        self.length = problem.meta_data.n_variables
        self.solution = []
        self.fitness = None

    def construct_solution(self, pheromones, alpha=1.0, beta=1.0):
        self.solution = []
        for i in range(self.length):
            tau_0 = pheromones[i][0]
            tau_1 = pheromones[i][1]

            eta_0 = 1.0
            eta_1 = 1.5

            prob_0 = (tau_0 ** alpha) * (eta_0 ** beta)
            prob_1 = (tau_1 ** alpha) * (eta_1 ** beta)

            total = prob_0 + prob_1
            prob_1_normalised = prob_1 / total

            if random.random() < prob_1_normalised:
                self.solution.append(1)
            else:
                self.solution.append(0)

    def evaluate(self):
        self.fitness = self.problem(self.solution)
        return self.fitness


class ACO:
    def __init__(self, problem: ProblemType, population_size=10, generation_count=100000, alpha=0.5, beta=2.0):
        self.problem = problem
        if self.problem.meta_data.problem_id == 18:
            self.optimum: int = 8
        else:
            self.optimum: int = self.problem.optimum.y

        self.population_size = population_size
        self.generation_count = generation_count
        self.ants = []
        self.pheromones = np.ones((problem.meta_data.n_variables, 2))
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = 0.1

        self.best_solution = None
        self.best_fitness = -np.inf

        for i in range(population_size):
            ant = Ant(self.problem)
            ant.construct_solution(self.pheromones, self.alpha, self.beta)
            self.ants.append(ant)

    def apply_pheromone_update(self):
        # Evaporation
        for i in range(len(self.pheromones)):
            self.pheromones[i][0] = (1 - self.evaporation_rate) * self.pheromones[i][0]
            self.pheromones[i][1] = (1 - self.evaporation_rate) * self.pheromones[i][1]

        # Deposition through elitism
        num_ants = int(0.1 * self.population_size)
        best_ants = sorted(self.ants, key=lambda ant: ant.evaluate(), reverse=True)[0:num_ants]
        
        # for ant in best_ants:
        #     print(ant.fitness)
        
        weighted = 1 / num_ants
        for ant in best_ants:
            deposit_amount = (ant.fitness / self.best_fitness) * weighted
            for i, bit in enumerate(ant.solution):
                self.pheromones[i][bit] += deposit_amount

        self.pheromones = np.clip(self.pheromones, 0.01, 10.0)
    
    def local_search(self, ant: Ant):
        improved = True
        while improved:
            improved = False
            for i in range(len(ant.solution)):
                local_ant = ant.solution.copy()
                local_ant[i] = 1 - local_ant[i]
                fitness = self.problem(local_ant)
                
                if fitness > ant.fitness:
                    ant.solution = local_ant
                    ant.fitness = fitness
                    improved = True

    def run(self):
        for generation in range(self.generation_count):
            if self.best_fitness >= self.optimum:
                break

            for ant in self.ants:
                ant.construct_solution(self.pheromones, self.alpha, self.beta)
                ant.evaluate()
                if random.random() < 0.2:
                    self.local_search(ant)

                if ant.fitness > self.best_fitness:
                    self.best_fitness = ant.fitness
                    self.best_solution = ant.solution.copy()

            self.apply_pheromone_update()

            if generation % 50 == 0:
                best_ant = max(self.ants, key=lambda ant: ant.fitness)
                print(f"Gen {generation}: Best fitness = {best_ant.fitness}")
                print(f"Best solution sample: {best_ant.solution[:20]}...")
                for i in range(8):
                    for j in range(8):
                        print(best_ant.solution[j + (i * i)], end='')
                    print()
                
            # if generation % 200 == 0:
            #     print(self.pheromones)
                
            if generation % 10000 == 0:
                self.pheromones = np.ones((self.problem.meta_data.n_variables, 2))

        print(f"found in generation: {generation}")
        return self.best_fitness, self.best_solution
