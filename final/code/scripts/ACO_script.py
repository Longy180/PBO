from ioh import get_problem, ProblemClass, logger
from ACO import ACO

if __name__ == '__main__':
    problems = []
    num_dim = 100
    f = [1, 2, 3, 18, 23, 24, 25]

    # Create all problem instances
    for i in f:
        problems.append(get_problem(fid=i, dimension=num_dim, instance=1, problem_class=ProblemClass.PBO))

    # Your ACO uses generation_count, not budget
    # Each generation evaluates population_size ants
    # Plus local search does ~10% * population_size * n additional evaluations
    # To approximate 500,000 evaluations with population_size=50:
    # 500,000 / 50 = 10,000 generations (ignoring local search overhead)
    
    population_size = 50
    generation_count = 10000  # Approximately 500k evaluations

    print(f"\n{'='*60}")
    print(f"Running: Custom ACO Algorithm (Exercise 5)")
    print(f"Population size: {population_size}, Generations: {generation_count}")
    print(f"{'='*60}")
    
    # Create logger for your custom ACO
    l = logger.Analyzer(
        root="data",
        folder_name="CustomACO",
        algorithm_name="CustomACO",
        algorithm_info="Custom ACO with 50 ants and local search"
    )
    
    # Run on all problems
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Problem {problem.meta_data.problem_id}: {problem.meta_data.name}")
        print(f"{'='*60}")
        problem.attach_logger(l)
        
        # Run 10 independent runs per problem
        for run in range(10):
            print(f"\nRun {run+1}/10:")
            
            # Create fresh ACO instance for this run
            aco = ACO(
                problem=problem,
                population_size=population_size,
                generation_count=generation_count,
                alpha=1.0,
                beta=1.0
            )
            
            # Run the algorithm
            best_fitness, best_solution = aco.run()
            
            print(f"  Final best fitness: {best_fitness}")
            print(f"  Optimum: {problem.optimum.y}")
            
            problem.reset()
    
    # Close logger
    del l
    
    print(f"\n{'='*60}")
    print("Custom ACO completed!")
    print("Data saved in: data/CustomACO/")
    print("Next: Zip data/CustomACO/ together with data/MMAS_* folders")
    print("      Upload to IOHanalyzer for comparison plots")
    print(f"{'='*60}\n")