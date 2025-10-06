from genetic import GeneticAlgorithm
from ioh import get_problem, ProblemClass

if __name__ == "__main__":
    problem = get_problem(fid = 25, dimension = 100, instance = 1, problem_class = ProblemClass.PBO)
    print(f"The problem name is: {problem.meta_data.name}")
    print(f"The problem optimum is: {problem.optimum.y}\n")