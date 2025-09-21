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

    def construct_solution(self, pheromones, heuristic, alpha=1.0, beta=2.0):
        self.solution = []
        for i in range(self.length):
            tau_0 = pheromones[i][0]
            tau_1 = pheromones[i][1]

            # Heuristic
            eta_0 = heuristic[0]
            eta_1 = heuristic[1]

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
    def __init__(self, problem: ProblemType, population_size=50, generation_count=100000, alpha=1.0, beta=1.0):
        self.problem = problem
        if self.problem.meta_data.problem_id == 18:
            self.optimum: int = 8
        else:
            self.optimum: int = self.problem.optimum.y

        self.population_size = population_size
        self.generation_count = generation_count
        self.ants = []

        self.pheromones = []
        for i in range(problem.meta_data.n_variables):
            self.pheromones.append([1.0, 1.0])
        
        # print(f"initial pheromones: {self.pheromones}\n")

        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = 0.2

        self.best_solution = None
        self.best_fitness = -np.inf

        self.heuristic_dict = {1: [0.9, 1.1], 
                          2: [0.9, 1.1], 
                          3: [0.9, 1.1], 
                          18: [1.0, 1.0], 
                          23: [1.0, 1.0], 
                          24: [1.0, 1.0], 
                          25: [1.0, 1.0]}
        
        self.heuristic = self.heuristic_dict[self.problem.meta_data.problem_id]

        for i in range(population_size):
            ant = Ant(self.problem)
            ant.construct_solution(self.pheromones, self.heuristic, self.alpha, self.beta)
            self.ants.append(ant)

    def pheromone_update(self):
        # Evaporation
        for i in range(len(self.pheromones)):
            self.pheromones[i][0] *= (1 - self.evaporation_rate)
            self.pheromones[i][1] *= (1 - self.evaporation_rate)

        # Get elite ants
        num_elite = max(1, int(0.2 * self.population_size))
        best_ants = sorted(self.ants, key=lambda ant: ant.fitness, reverse=True)[:num_elite]  # Highest first

        fitnesses = [ant.fitness for ant in self.ants]
        min_fit = min(fitnesses)
        max_fit = max(fitnesses)

        weighted = 1 / num_elite
        for ant in best_ants:
            norm_fit = (ant.fitness - min_fit + 1) / (max_fit - min_fit + 1)
            deposit_amount = norm_fit * weighted
            for i, bit in enumerate(ant.solution):
                self.pheromones[i][bit] += deposit_amount

                # Apply reasonable bounds
        self.pheromones = np.clip(self.pheromones, 0.1, 10.0)
    
    def local_search(self, ant: Ant):
        current_fitness = ant.fitness
        current_solution = ant.solution.copy()

        for i in range(ant.length):
            neighbour = ant.solution.copy()
            neighbour[i] = 1 - neighbour[i]
            fitness = self.problem(neighbour)
            
            if fitness > current_fitness:
                current_fitness = fitness
                current_solution = neighbour
        
        ant.solution = current_solution
        ant.fitness = current_fitness

    def run(self):
        for generation in range(self.generation_count):
            if self.best_fitness >= self.optimum:
                break

            for ant in self.ants:
                ant.construct_solution(self.pheromones, self.heuristic, self.alpha, self.beta)
                ant.evaluate()
                if random.random() < 0.1:
                    self.local_search(ant)

                # print(f"fitness: {ant.fitness}")
                if ant.fitness > self.best_fitness:
                    self.best_fitness = ant.fitness
                    self.best_solution = ant.solution.copy()

            self.pheromone_update()

            if generation % 2000 == 0:
                best_ant = max(self.ants, key=lambda ant: ant.fitness)
                print(f"Gen {generation}: Best fitness = {best_ant.fitness}")
                # print(f"Best solution sample: {best_ant.solution[:20]}...")
                # for i in range(8):
                #     for j in range(8):
                #         print(best_ant.solution[j + (i * 8)], end='')
                #     print()
                # print("pheromones")
                # print(self.pheromones)

            # if generation % 30 == 0 and generation != 0:
            #     print("pheromones")
            #     print(self.pheromones)
            #     exit()

        print(f"found in generation: {generation}")
        return self.best_fitness, self.best_solution