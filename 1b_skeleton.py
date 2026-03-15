"""
This is a skeleton file for the implementation of your Genetic Algorithm (GA).
You can add functions to the file and implement your logic inside genetic_algorithm(),
but do not modify the other functions. print_generation_info() must be called on each iteration of the GA.
"""

# <--- ADD ADDITONAL IMPORTS HERE --->
import argparse




# <---------------------------------->


# <--- ADD ADDITONAL DEFINES HERE --->

# Population size: Defines how many individuals are in the initial population. (You can change this value)
POPULATION_SIZE = 100 # use this value for the generation of your inital population.




# <---------------------------------->


# <--- ADD ADDITONAL FUNCTIONS HERE --->





# <------------------------------------>

def genetic_algorithm(gui_mode=False):
    """
    Implementation of your genetic algorithm.

    Args:
        gui_mode (bool): If True, run the algorithm with a GUI. Is completly free to you if you want to use that.
    """

    pass  # TODO: Implement GA logic

def print_generation_info(generation: int, best_fitness: float, mean_fitness: float) -> None:
    """
    Displays the statistics of the current population in a structured format.

    Args:
        generation (int): The current generation number.
        best_fitness (float): The best fitness value in the current population.
        mean_fitness (float): The arithmetic mean (average) fitness value of the current population.
    """

    N = 100  # Print every 100 generations (adjustable)
    if generation % N == 0:
        print(f" Generation {generation:>7} | Best Fitness: {best_fitness:.2f} | Mean Fitness: {mean_fitness:.2f} ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Genetic Algorithm for solving the 1024-Queens Problem.")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode for visualization.")

    genetic_algorithm(gui_mode=parser.parse_args().gui)
