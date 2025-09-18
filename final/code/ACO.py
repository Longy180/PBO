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

            # Heuristic
            eta_0 = 1.0
            eta_1 = 2.0

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

        self.pheromones = []
        for i in range(problem.meta_data.n_variables):
            self.pheromones.append([0.5, 0.5])
        
        print(f"initial pheromones: {self.pheromones}\n")

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
            self.pheromones[i][0] *= (1 - self.evaporation_rate)
            self.pheromones[i][1] *= (1 - self.evaporation_rate)

        # Get elite ants
        num_elite = max(1, int(0.2 * self.population_size))
        
        best_ants = sorted(self.ants, key=lambda ant: ant.fitness, reverse=True)[:num_elite]  # Highest first

        # Calculate deposits using rank-based method (MUCH more stable)
        for rank, ant in enumerate(best_ants):
            # Rank-based deposit (best gets most, worst gets least)
            rank_weight = (num_elite - rank) / num_elite
            deposit_amount = 0.3 * rank_weight  # Scale as needed
            
            print(f"Ant rank {rank}: fitness={ant.fitness:.1f}, deposit={deposit_amount:.3f}")
            
            # Apply deposits
            for i, bit in enumerate(ant.solution):
                self.pheromones[i][bit] += deposit_amount

        # Apply reasonable bounds
        self.pheromones = np.clip(self.pheromones, 0.01, 10.0)
    
    def local_search(self, ant: Ant):
        random_index = random.randint(0, ant.length - 1)
        local_ant = ant.solution.copy()
        local_ant[random_index] = 1 - local_ant[random_index]
        fitness = self.problem(local_ant)
        
        if fitness > ant.fitness:
            ant.solution = local_ant
            ant.fitness = fitness

    def run(self):
        for generation in range(self.generation_count):
            if self.best_fitness >= self.optimum:
                break

            for ant in self.ants:
                ant.construct_solution(self.pheromones, self.alpha, self.beta)
                ant.evaluate()
                self.local_search(ant)

                print(f"fitness: {ant.fitness}")
                if ant.fitness > self.best_fitness:
                    self.best_fitness = ant.fitness
                    self.best_solution = ant.solution.copy()

            self.apply_pheromone_update()

            if generation % 2000 == 0:
                best_ant = max(self.ants, key=lambda ant: ant.fitness)
                print(f"Gen {generation}: Best fitness = {best_ant.fitness}")
                print(f"Best solution sample: {best_ant.solution[:20]}...")
                for i in range(8):
                    for j in range(8):
                        print(best_ant.solution[j + (i * i)], end='')
                    print()
                print("pheromones")
                print(self.pheromones)

            if generation % 2000 == 0 and generation != 0:
                exit()

        print(f"found in generation: {generation}")
        return self.best_fitness, self.best_solution
    
    # Quick test to verify OneMax behavior
    def test_fitness(self, problem):
        """Test that OneMax works as expected"""
        
        print(f"Fitness values for {problem.meta_data.name}\n")
        # Test case 1: All zeros
        all_zeros = [0] * problem.meta_data.n_variables
        fitness_zeros = problem(all_zeros)
        print(f"All zeros fitness: {fitness_zeros}")
        
        # Test case 2: All ones  
        all_ones = [1] * problem.meta_data.n_variables
        fitness_ones = problem(all_ones)
        print(f"All ones fitness: {fitness_ones}")
        
        # Test case 3: Half ones
        half_ones = [1 if i < problem.meta_data.n_variables//2 else 0 
                    for i in range(problem.meta_data.n_variables)]
        fitness_half = problem(half_ones)
        print(f"Half ones fitness: {fitness_half}")
        
        # Test case 4: Manual count
        manual_count = sum(half_ones)
        print(f"Manual count of half_ones: {manual_count}")
        
        print(f"Problem optimum: {problem.optimum.y if hasattr(problem, 'optimum') else 'Unknown'}")
        print(f"Problem variables: {problem.meta_data.n_variables}")