# from ioh import get_problem, ProblemType, ProblemClass, logger
# import sys
# import numpy as np
# import random
# import math

# # MMAS
# def mmas(problem: ProblemType, budget: int | None = None) -> tuple[float,list[int]]:
#     # Setup budget = 50n^2
#     if budget is None:
#         budget = int(problem.meta_data.n_variables * problem.meta_data.n_variables * 50)

#     # Setup Problem Optimum
#     if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
#         optimum: int = 8
#     else:
#         optimum: int = problem.optimum.y
    
#     # Setup x & f & path
#     f_opt: float = float('-inf')
#     length: int = problem.meta_data.n_variables
#     path = [[0.5,0.5]]*length
#     p_options = [1,1/length,1/math.sqrt(length)]
#     p = p_options[0] # p
#     x_opt: list[int] = list(np.random.randint(length, size = problem.meta_data.n_variables))
#     for _ in range(budget):
#         # simulate ant movement through binary path
#         my_path = []
#         for i in range(length):
#             total_probability = path[i][0]+path[i][1]
#             if random.uniform(0,total_probability) > path[i][0]:
#                 my_path.append(1)
#             else:
#                 my_path.append(0)

#         f = problem(my_path)
#         # If better OR EQUAL than optimum then update optimum & Path pheremones
#         if f >= f_opt:
#             f_opt = f
#             x_opt = my_path
#             for i in range(len(my_path)):
#                 bit_path = [0,0]
#                 bit_path[my_path[i]] = min((1-p)*path[i][my_path[i]]+p,1-1/length)#included
#                 bit_path[1-my_path[i]] = max((1-p)*path[i][1-my_path[i]],1/length)#not included
#                 path[i] = bit_path
        
#         # If better than problem optimum then return.
#         if f_opt >= optimum:
#             print("done")
#             break
#     return f_opt, x_opt

# # MMAS*
# def mmasStar(problem: ProblemType, budget: int | None = None) -> tuple[float,list[int]]:
#     # Set budget to 50n^2    
#     if budget is None:
#         budget = int(problem.meta_data.n_variables * problem.meta_data.n_variables * 50)

#     # Setup Problem Optimum
#     if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
#         optimum: int = 8
#     else:
#         optimum: int = problem.optimum.y
    
#     # Setup x & f & path
#     f_opt: float = float("-inf")
#     length: int = problem.meta_data.n_variables
#     x_opt: list[int] = list(np.random.randint(length, size = problem.meta_data.n_variables))
#     path = [[0.5,0.5]]*length
#     p_options = [1,1/length,1/math.sqrt(length)]
#     p = p_options[0] # p
#     for _ in range(budget):
#         # Simulate ant movement through binary path
#         my_path = []
#         for i in range(length):
#             total_probability = path[i][0]+path[i][1]
#             if random.uniform(0,total_probability) > path[i][0]:
#                 my_path.append(1)
#             else:
#                 my_path.append(0)

#         f = problem(my_path)
#         # If better than optimum then update optimum & path pheremones
#         if f > f_opt:
#             f_opt = f
#             x_opt = my_path
#             for i in range(len(my_path)):
#                 bit_path = [0,0]
#                 bit_path[my_path[i]] = min((1-p)*path[i][my_path[i]]+p,1-1/length)#included
#                 bit_path[1-my_path[i]] = max((1-p)*path[i][1-my_path[i]],1/length)#not included
#                 path[i] = bit_path
        
#         # If better than problem optimum then return
#         if f_opt >= optimum:
#             print("done")
#             break;
#     return f_opt, x_opt


##### MMAS Script #####

from ioh import ProblemType
import numpy as np
import random

# MMAS
def mmas(problem: ProblemType, rho: float, budget: int | None = None) -> tuple[float, list[int]]:
    """
    Max-Min Ant System algorithm
    
    Args:
        problem: IOH problem instance
        rho: Evaporation rate (ρ parameter)
        budget: Maximum number of fitness evaluations
    
    Returns:
        Tuple of (best_fitness, best_solution)
    """
    # Setup budget = 50n^2
    if budget is None:
        budget = int(problem.meta_data.n_variables * problem.meta_data.n_variables * 50)

    # Setup Problem Optimum
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum: int = 8
    else:
        optimum: int = problem.optimum.y
    
    # Setup x & f & path
    f_opt: float = float('-inf')
    length: int = problem.meta_data.n_variables
    path = [[0.5, 0.5]] * length
    p = rho  # Use the passed rho parameter
    x_opt: list[int] = list(np.random.randint(2, size=problem.meta_data.n_variables))
    
    for _ in range(budget):
        # Simulate ant movement through binary path
        my_path = []
        for i in range(length):
            total_probability = path[i][0] + path[i][1]
            if random.uniform(0, total_probability) > path[i][0]:
                my_path.append(1)
            else:
                my_path.append(0)

        f = problem(my_path)
        
        # If better OR EQUAL than optimum then update optimum & Path pheromones
        if f >= f_opt:
            f_opt = f
            x_opt = my_path
            for i in range(len(my_path)):
                bit_path = [0, 0]
                bit_path[my_path[i]] = min((1 - p) * path[i][my_path[i]] + p, 1 - 1/length)  # included
                bit_path[1 - my_path[i]] = max((1 - p) * path[i][1 - my_path[i]], 1/length)  # not included
                path[i] = bit_path
        
        # If better than problem optimum then return
        if f_opt >= optimum:
            break
            
    return f_opt, x_opt


# MMAS*
def mmasStar(problem: ProblemType, rho: float, budget: int | None = None) -> tuple[float, list[int]]:
    """
    Max-Min Ant System* algorithm (strict improvement only)
    
    Args:
        problem: IOH problem instance
        rho: Evaporation rate (ρ parameter)
        budget: Maximum number of fitness evaluations
    
    Returns:
        Tuple of (best_fitness, best_solution)
    """
    # Set budget to 50n^2    
    if budget is None:
        budget = int(problem.meta_data.n_variables * problem.meta_data.n_variables * 50)

    # Setup Problem Optimum
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum: int = 8
    else:
        optimum: int = problem.optimum.y
    
    # Setup x & f & path
    f_opt: float = float("-inf")
    length: int = problem.meta_data.n_variables
    x_opt: list[int] = list(np.random.randint(2, size=problem.meta_data.n_variables))
    path = [[0.5, 0.5]] * length
    p = rho  # Use the passed rho parameter
    
    for _ in range(budget):
        # Simulate ant movement through binary path
        my_path = []
        for i in range(length):
            total_probability = path[i][0] + path[i][1]
            if random.uniform(0, total_probability) > path[i][0]:
                my_path.append(1)
            else:
                my_path.append(0)

        f = problem(my_path)
        
        # If STRICTLY better than optimum then update optimum & path pheromones
        if f > f_opt:
            f_opt = f
            x_opt = my_path
            for i in range(len(my_path)):
                bit_path = [0, 0]
                bit_path[my_path[i]] = min((1 - p) * path[i][my_path[i]] + p, 1 - 1/length)  # included
                bit_path[1 - my_path[i]] = max((1 - p) * path[i][1 - my_path[i]], 1/length)  # not included
                path[i] = bit_path
        
        # If better than problem optimum then return
        if f_opt >= optimum:
            break
            
    return f_opt, x_opt