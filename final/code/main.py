from ioh import get_problem, ProblemType, ProblemClass, logger
import sys
import numpy as np
import optimisationAlgorithms
from mmas import mmas, mmasStar
from ACO import ACO

def random_search(func, iterations, budget = None) -> tuple[float, list[int]]:
    # budget of each run: 50n^2
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y
    #print(optimum)
    # 10 independent runs for each algorithm on each problem.
    f_opt: float = sys.float_info.min
    x_opt: list[int] = list(np.random.randint(2, size = func.meta_data.n_variables))
    for r in range(iterations):
        f_opt = sys.float_info.min
        x_opt = list(np.random.randint(2, size = func.meta_data.n_variables))
        for i in range(budget):
            x: list[int] = list(np.random.randint(2, size = func.meta_data.n_variables))
            f: float = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
        func.reset()
    return f_opt, x_opt


if __name__ == '__main__':

    func = optimisationAlgorithms.ea11

    problems = []
    num_dim = 100
    f = [18, 23, 24, 25]

    for i in f:
        problems.append(get_problem(fid = i, dimension = num_dim, instance = 1, problem_class = ProblemClass.PBO))

    l = logger.Analyzer(root = "data",
    folder_name = "run",
    algorithm_name=func.__name__,
    algorithm_info="test of IOHExperimenter in python")
    print(func.__name__)

    for problem in problems:
        print(f"Problem {problem.meta_data.problem_id}: {problem.meta_data.name}\n")
        aco = ACO(problem=problem)
        problem.attach_logger(l)
        budget = 100000
        print(f"found sol: {aco.run()} | sol optimum: {problem.optimum.y} | mmas sol: {mmasStar(problem, budget)[0]}\n") #,random_search(problem,budget)[0])