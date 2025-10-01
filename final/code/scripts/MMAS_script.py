from ioh import get_problem, ProblemClass, logger
import math
from mmas import mmas, mmasStar

if __name__ == '__main__':
    problems = []
    num_dim = 100
    f = [1, 2, 3, 18, 23, 24, 25]

    for i in f:
        problems.append(get_problem(fid=i, dimension=num_dim, instance=1, problem_class=ProblemClass.PBO))

    # Define all 6 configurations
    configs = [
        ("MMAS_rho1", mmas, 1),
        ("MMAS_rho_inv_n", mmas, 1/num_dim),
        ("MMAS_rho_inv_sqrt_n", mmas, 1/math.sqrt(num_dim)),
        ("MMAS_star_rho1", mmasStar, 1),
        ("MMAS_star_rho_inv_n", mmasStar, 1/num_dim),
        ("MMAS_star_rho_inv_sqrt_n", mmasStar, 1/math.sqrt(num_dim))
    ]

    budget = 500000  # 50 * 100^2

    # Run each configuration
    for config_name, algorithm, rho in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config_name} with ρ={rho}")
        print(f"{'='*60}")
        
        # Create separate logger for this configuration
        l = logger.Analyzer(
            root="data",
            folder_name=config_name,  # Separate folder for each
            algorithm_name=config_name,
            algorithm_info=f"MMAS with rho={rho}"
        )
        
        # Run on all problems
        for problem in problems:
            print(f"Problem {problem.meta_data.problem_id}: {problem.meta_data.name}")
            problem.attach_logger(l)
            
            # Run 10 independent runs
            for run in range(10):
                result = algorithm(problem, rho, budget)
                print(f"  Run {run+1}: {result[0]}")
                problem.reset()
        
        # Close logger
        del l
        print(f"Completed {config_name}\n")