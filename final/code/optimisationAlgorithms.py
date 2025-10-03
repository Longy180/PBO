from typing import Sequence
from ioh import get_problem, ProblemType, ProblemClass, logger
import sys
import numpy as np
import random

# EA(1+1) Algorithm
def ea11(problem: ProblemType, budget: int = 100000) -> tuple[float,list[int]]:
    # Set budget to 50n^2
    if budget is None:
        budget = int(problem.meta_data.n_variables * problem.meta_data.n_variables * 50)
    # setup Problem Optimum
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum: int = 8
    else:
        optimum: int = problem.optimum.y

    # Setup x & f
    f_opt: float = float("-inf")
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
def RLS(problem: ProblemType, budget: int = 100000, n: int = 100) -> tuple[float,list[int]]:
    # setup Problem Optimum
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum: int = 8
    else:
        optimum: int = problem.optimum.y
    
    # Setup x & f
    f_opt: float = float("-inf")
    x_opt: list[int] = []
    for i in range(n):
        x_opt.append(random.randint(0, 1))
    
    for i in range(budget):
        x: list[int] = x_opt.copy()

        # Flip one randomly chosen bit
        randFlip: int = random.randint(0, n - 1)
        x[randFlip] = 1 - x[randFlip]
        f: float = problem(x)

        # If better than optimum update optimum
        if f > f_opt:
            f_opt = f
            x_opt = x

        # if better than problem optimum then return.
        if f_opt >= optimum:
            print(f"done in {i} iterations.")
            break

    return f_opt, x_opt

def random_search(func, budget = 100000):
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

problem_ids = [1, 2, 3, 18, 23, 24, 25]
dimension = 100
budget = 100000
num_runs = 10

algorithms = [
    (random_search, "RandomSearch"),
    (RLS, "RLS"),
    (ea11, "EA_1plus1")
]

# Run experiments for each algorithm across all problems
for algorithm, algorithm_name in algorithms:

    l = logger.Analyzer(
        root="/Users/axvd/Downloads/", 
        folder_name="run",
        algorithm_name=algorithm_name, 
        algorithm_info=f"Benchmark of {algorithm_name}"
    )
    
    # Run this algorithm on all problems
    for fid in problem_ids:
        
        # Create problem instance
        problem = get_problem(
            fid=fid, 
            dimension=dimension, 
            instance=1, 
            problem_class=ProblemClass.PBO
        )

        problem.attach_logger(l)
        
        # Run algorithm 10 times on this problem
        for run in range(num_runs):
            algorithm(problem, budget=budget)
            problem.reset() 
        
    
    # Delete logger after this algorithm completes all problems
    del l
