"""
This is a skeleton file for the implementation of your Genetic Algorithm (GA).
You can add functions to the file and implement your logic inside genetic_algorithm(),
but do not modify the other functions. print_generation_info() must be called on each iteration of the GA.
"""

# <--- ADD ADDITONAL IMPORTS HERE --->
import argparse
import numpy as np
import os
from numba import njit

#TODO: if best fitness doesn't change over multiple generations, add some stronger variation

# <---------------------------------->


# <--- ADD ADDITONAL DEFINES HERE --->

# Population size: Defines how many individuals are in the initial population. (You can change this value)
MAX_GENERATIONS = 1000 # max. repetitions of starting off with slightly altered population
POPULATION_SIZE = 100 # size of the initial populatioon (number of individual queen configurations)
NUM_QUEENS = 512 # number of queens (and their positions) in one individual
BEST_NUM = POPULATION_SIZE//10 # number of individuals taken over unaltered to the next generation
RAND_RATIO = 0.5   #ratio of best individuals used for randomization
CROSS = True     #mixing of two random individuals within RAND_RATIO of best ones
MUTATE = True    #switching MUTATION_NUM of queens in one of the RAND_RATIO best individuals
MUTATION_NUM = 2 #number of columns (queen positions) swapped in mutation

#possible moves of knight relative to its position
relative_knight_moves = np.array([(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)],dtype=np.int32)

#using previously found good population with good maximum fit (to not start off completely from zero)
LOAD_BEST = True
SAVE_BEST = True
save_folder = "./prior_populations"
#load_path = f"{save_folder}/best_solution_6.npy"
load_path = f"1b_working_solution.npy"
working_path = "1b_working_solution.npy"

#heavier randomization if fitness doesn't improve after MAX_STATNATION generations
STAGNATION_DETECTION = True
MAX_STAGNATION = 10 #max. allowed subsequent generations with same fitness
RESET_RATIO = 2    #ratio of (worst) individuals that will be completely randomized again

# <---------------------------------->


# <--- ADD ADDITONAL FUNCTIONS HERE --->
def generate_population(pop_size):
    pop = []
    #initialization of all queens in separate rows and columns
    for _ in range(pop_size):
        pop.append(np.random.permutation(NUM_QUEENS))
    print(f"Initialized random population of size {POPULATION_SIZE} with {NUM_QUEENS} queens each.")
    return pop

@njit
def get_fitness(pop):
    safe = [1]*512

    #efficient diagonal check
    #diagonals have a constant row+col or row-col, so values lie between 0 and 511*2
    diagonal_1 = np.zeros(2*NUM_QUEENS) 
    diagonal_2 = np.zeros(2*NUM_QUEENS)

    #count the queens per diagonal
    for col in range(NUM_QUEENS):
        row = int(pop[col])
        #diagonals are defined by having a constant value for row-col
        # i.e. main falling diagonal row-col = 1, first diagonal above that has row-col = -1
        #      main rising diagonal  row+col = 512, first diagonal above that has row+col = 512 -1
        d1 = int(row-col + NUM_QUEENS) # keep the value positive for convenience
        d2 = int(row+col)
        diagonal_1[d1] += 1
        diagonal_2[d2] += 1

    #mark where queens are on same diagonal
    for col in range(NUM_QUEENS):
        row = int(pop[col])

        #check on which rising and falling diagonal the queen is located
        d1 = int(row-col + NUM_QUEENS)
        d2 = int(row+col)
        #if another queen is on that diagonal, queen is threatened
        if diagonal_1[d1] > 1 or diagonal_2[d2] > 1:
            safe[col] = 0

    for col in range(NUM_QUEENS):
        row = int(pop[col])

        #attacks from knight moves
        for kmx, kmy in relative_knight_moves:
            col_threat = int(col+kmy)
            row_threat = int(row+kmx)

            if (0<= col_threat < NUM_QUEENS) and (0<= row_threat <NUM_QUEENS):
                if pop[col_threat] == row_threat:
                    safe[col] = safe[col_threat] = 0
    
    #return number of safe queens
    return sum(safe)

def get_best_individuals(pop,fitnesses,sel = BEST_NUM):
    #sorting individuals based on best fitnesses
    ranked_fitnesses, ranked_pop = list(zip(*sorted(zip(fitnesses,pop),key = lambda x: x[0],reverse=True)))

    return list(ranked_pop[:sel])

def cross(ind1,ind2):
    #crossing two individuals without causing any queens to be in the same row
    a, b = sorted(np.random.choice(range(NUM_QUEENS), 2, replace=False))

    pop_new = np.ones(NUM_QUEENS)*-1#initialize with invalid values

    # choose random segment out of individual 1
    pop_new[a:b] = ind1[a:b]
    # fill in only the remaining rows with quees from individual 2
    ind2_remaining = [ind2[i] for i in range(NUM_QUEENS) if (ind2[i] not in ind1[a:b])]
    for p,pn in enumerate(pop_new):
        if pn == -1:
            pop_new[p] = ind2_remaining.pop(0)

    return np.array(pop_new)

def mutate(ind1, number):
    
    #saving original version, to see if mutation made it better or worse
    ind1_original = ind1.copy()
    fitness_original = get_fitness(ind1_original)

    #randomly perform number of column swaps, so no row conflicts are created
    #for now duplicate swaps are possible
    for _ in range(number):
        col1,col2 = np.random.choice(NUM_QUEENS, 2, replace=False)
        ind1[col2], ind1[col1] = ind1[col1], ind1[col2]
    
    fitness_new = get_fitness(ind1)

    #only accept the mutation if it improved fitness
    if fitness_new>fitness_original:
        return ind1
    else:
        return ind1_original

    


# <------------------------------------>

def genetic_algorithm(gui_mode=False):
    """
    Implementation of your genetic algorithm.

    Args:
        gui_mode (bool): If True, run the algorithm with a GUI. Is completly free to you if you want to use that.
    """
    current_pop = generate_population(POPULATION_SIZE)
    best_fitness = 0
    mean_fitness = 0
    generation = 0

    stagnation_counter = 0
    last_best = 0

    #load the prevously found best version to not start from scratch
    try:
        if os.path.exists(load_path) and LOAD_BEST:
            saved = np.load(load_path)
            # if the file holds a permutation of the right size, include it
            if isinstance(saved, np.ndarray) and saved.shape == (NUM_QUEENS,):
                current_pop[0] = saved.copy()
                print(">>> Loaded previous solution into initial population")
    except Exception:
        # ignore any loading errors
        pass

    while best_fitness<NUM_QUEENS and generation<MAX_GENERATIONS:

        #evaluate individuals statistics in current population
        fitnesses = [get_fitness(ind) for ind in current_pop]
            #print(f"Checking individual {k+1}/{POPULATION_SIZE}",end="\r")
        mean_fitness = sum(fitnesses)/POPULATION_SIZE
        if max(fitnesses)>best_fitness:
            best_fitness = max(fitnesses)
        #print(f"")
        print_generation_info(generation,best_fitness,mean_fitness)

        #choose the best individuals of the current population (in this case all are chosen, but now ordered by fitness)
        ordered_pop = get_best_individuals(current_pop,fitnesses,POPULATION_SIZE)

        #add best of prevous population, but only a few to not get stuck in local maxima
        new_population = ordered_pop[:BEST_NUM]

        #fill up new population
        while len(new_population) < POPULATION_SIZE:

            #add slightly altered versions of previous population
            rand_choice = np.random.randint(POPULATION_SIZE)
            best_ind_altered = ordered_pop[rand_choice].copy()#in case no crossing is used
            if CROSS:
                rand_ind1 = ordered_pop[np.random.randint(int(POPULATION_SIZE*RAND_RATIO))]
                rand_ind2 = ordered_pop[np.random.randint(int(POPULATION_SIZE*RAND_RATIO))]
                best_ind_altered = cross(rand_ind1,rand_ind2)
            if MUTATE:
                best_ind_altered = mutate(best_ind_altered,MUTATION_NUM)
            new_population.append(best_ind_altered)

        #check if progress is stagnating -> add stronger variation if necessary
        if best_fitness > last_best:
            stagnation_counter = 0
            last_best = best_fitness
        else:
            stagnation_counter += 1
        
        #randomize part of population again in case progress is stagnating
        if stagnation_counter > MAX_STAGNATION and STAGNATION_DETECTION:
            print(">>> Stagnation detected – injecting diversity")

            for i in range(POPULATION_SIZE//RESET_RATIO, POPULATION_SIZE):
                new_population[i] = np.random.permutation(NUM_QUEENS)

            stagnation_counter = 0
        
        #use newly shuffled population
        current_pop = new_population
        generation += 1

        
        #save a copy of the best solution every time to not loose progress
        best_solution = current_pop[np.argmax(fitnesses)].copy()
        if (best_solution is not None) and SAVE_BEST:
            try:
                np.save(working_path, best_solution)
                #print(f"Saved best solution (fitness={best_fitness:.2f}) to {working_path}")
            except Exception as e:
                print(f"Warning: could not save solution: {e}")


    

def print_generation_info(generation: int, best_fitness: float, mean_fitness: float) -> None:
    """
    Displays the statistics of the current population in a structured format.

    Args:
        generation (int): The current generation number.
        best_fitness (float): The best fitness value in the current population.
        mean_fitness (float): The arithmetic mean (average) fitness value of the current population.
    """

    N = 10  # Print every 100 generations (adjustable)
    if generation % N == 0:
        print(f" Generation {generation:>7} | Best Fitness: {best_fitness:.2f} | Mean Fitness: {mean_fitness:.2f} ")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Genetic Algorithm for solving the 1024-Queens Problem.")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode for visualization.")

    genetic_algorithm(gui_mode=parser.parse_args().gui)
