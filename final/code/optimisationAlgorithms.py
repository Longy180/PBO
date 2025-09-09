from ioh import get_problem, ProblemType, ProblemClass, logger
import sys
import numpy as np

# EA(1+1) Algorithm
def ea11(problem: ProblemType, iterations: int, budget: int | None = None) -> tuple[int,list[int]]:
    if budget is None:
        budget = int(problem.meta_data.n_variables * problem.meta_data.n_variables * 50)
    
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum: int = 8
    else:
        optimum: int = problem.optimum.y
    for _ in range(iterations):
        f_opt = sys.float_info.min
        x_opt: list[int] = np.random.randint(2, size = problem.meta_data.n_variables)
        for _ in range(budget):
            x: list[int] = x_opt.copy()
            rand = np.random.randint(len(x), size = problem.meta_data.n_variables)
            for i in range(len(x)):
                if rand[i] == 0:
                    x[i] = 1-x[i]
            f = problem(x)
            #print(f_opt,f,x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                print("done")
                break
            print(f)
        problem.reset()
    return f_opt, x_opt