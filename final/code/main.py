from ioh import get_problem, ProblemClass, logger
import sys
import numpy as np


def random_search(func, budget = None):
    # budget of each run: 50n^2
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y
    print(optimum)
    # 10 independent runs for each algorithm on each problem.
    for r in range(10):
        f_opt = sys.float_info.min
        x_opt = None
        for i in range(budget):
            x = np.random.randint(2, size = func.meta_data.n_variables)
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
        func.reset()
    return f_opt, x_opt


if __name__ == '__main__':

    func = random_search

    problems = []
    num_dim = 100
    f = [1,2,3,18,23,24,25]

    for i in f:
        problems.append(get_problem(fid = i, dimension = num_dim, instance = 1, problem_class = ProblemClass.PBO))


    l = logger.Analyzer(root = "data",
    folder_name = "run",
    algorithm_name=func.__name__,
    algorithm_info="test of IOHExperimenter in python")

    for problem in problems:
        problem.attach_logger(l)
        func(problem)
