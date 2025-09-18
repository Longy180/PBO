from genetic import GeneticAlgorithm
from ioh import get_problem, ProblemClass

if __name__ == "__main__":
    # Example run on OneMax (F1)
    problem = get_problem(fid = 23, dimension = 100, instance = 1, problem_class = ProblemClass.PBO)
    print(f"The problem name is: {problem.meta_data.name}")
    print(f"The problem optimum is: {problem.optimum.y}\n")
    ga = GeneticAlgorithm(pop_size=20, generations=100000)
    best = ga.run(problem)
    print("Best fitness found:", best.fitness)