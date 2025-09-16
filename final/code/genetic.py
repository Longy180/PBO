from population import Population
from individual import Individual
import numpy as np
from ioh import ProblemType

class GeneticAlgorithm:
    pop_size: int
    generations: int
    def __init__(self, pop_size=20, generations=1000):
        self.pop_size = pop_size
        self.generations = generations

    def run(self, problem: ProblemType) -> Individual | None:
        # setup Problem Optimum
        if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
            optimum: int = 8
        else:
            optimum: int = problem.optimum.y

        n: int = problem.meta_data.n_variables
        pop: Population = Population(self.pop_size, n, problem)
        best: Individual | None = pop.getBest()
        if best is None: return None

        for _ in range(self.generations):
            new_inds: list[Individual] = []
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
            elite: Individual | None = pop.getBest()
            if elite is None: return None
            pop.individuals = sorted(new_inds,
                                     key=lambda ind: ind.fitness, reverse=True)[:self.pop_size]
            popBest: Individual | None = pop.getBest()
            if popBest is None: return None
            if elite.fitness > popBest.fitness:
                pop.individuals[-1] = elite

            if popBest.fitness > best.fitness:
                best = pop.getBest()
                if best is None: return None
            
            if best.fitness >= optimum:
                print(f"done in {_} iterations.")
                break

        return best