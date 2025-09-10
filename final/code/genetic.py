from population import Population
import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size=20, generations=1000):
        self.pop_size = pop_size
        self.generations = generations

    def run(self, problem):
        # setup Problem Optimum
        if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
            optimum: int = 8
        else:
            optimum: int = problem.optimum.y

        n = problem.meta_data.n_variables
        pop = Population(self.pop_size, n, problem)
        best = pop.getBest()

        for _ in range(self.generations):
            new_inds = []
            for i in range(self.pop_size):
                # tournament selection
                parent1, parent2 = pop.informal_tournament_selection(k=3)
                child1, child2 = pop.performCrossover(parent1, parent2)
                child1.flipAllBitsMutation()
                child2.flipAllBitsMutation()
                child1.evaluate(problem)
                child2.evaluate(problem)
                new_inds.extend([child1, child2])

            # elitism: keep best of old + new
            elite = pop.getBest()
            pop.individuals = sorted(new_inds,
                                     key=lambda ind: ind.fitness, reverse=True)[:self.pop_size]

            if elite.fitness > pop.getBest().fitness:
                pop.individuals[-1] = elite

            if pop.getBest().fitness > best.fitness:
                best = pop.getBest()
            
            if best.fitness >= optimum:
                print(f"done in {_} iterations.")
                break

        return best