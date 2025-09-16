from ioh import get_problem, ProblemType, ProblemClass, logger
import sys
import numpy as np
import random

# EA(1+1) Algorithm
def ea11(problem: ProblemType, budget: int | None = None) -> tuple[int,list[int]]:
    # Set budget to 50n^2
    if budget is None:
        budget = int(problem.meta_data.n_variables * problem.meta_data.n_variables * 50)
    # setup Problem Optimum
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum: int = 8
    else:
        optimum: int = problem.optimum.y

    # Setup x & f
    f_opt = sys.float_info.min
    x_opt: list[int] = list(np.random.randint(2, size = problem.meta_data.n_variables))
    for _ in range(budget):
        x: list[int] = x_opt.copy()

        # Create list of probabilities for bit changes
        rand: list[int] = list(np.random.randint(len(x), size = problem.meta_data.n_variables))
        for i in range(len(x)):
            if rand[i] == 0:
                # Flip Bit
                x[i] = 1-x[i]
        f = problem(x)

        # If better than optimum update optimum
        if f > f_opt:
            f_opt = f
            x_opt = x

        # if better than problem optimum then return.
        if f_opt >= optimum:
            print(f"done in {_} iterations.")
            break

    return f_opt, x_opt

# RLS
def RLS(problem: ProblemType, budget=100000, n=100):
    # setup Problem Optimum
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum: int = 8
    else:
        optimum: int = problem.optimum.y
    
    # Setup x & f
    f_opt = sys.float_info.min
    x_opt = []
    for i in range(n):
        x_opt.append(random.randint(0, 1))
    
    for i in range(budget):
        x = x_opt.copy()

        # Flip one randomly chosen bit
        randFlip = random.randint(0, n - 1)
        x[randFlip] = 1 - x[randFlip]
        f = problem(x)

        # If better than optimum update optimum
        if f > f_opt:
            f_opt = f
            x_opt = x

        # if better than problem optimum then return.
        if f_opt >= optimum:
            print(f"done in {i} iterations.")
            break

    return f_opt, x_opt