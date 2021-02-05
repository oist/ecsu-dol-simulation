"""
TODO: Missing module docstring
"""

import os
import argparse
from pytictoc import TicToc
import numpy as np
from pyevolver.evolution import Evolution
from dol import gen_structure
from dol import utils
from dol.simulation import Simulation



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run the Division of Labor Simulation'
    )

    # evolution arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed')     
    parser.add_argument('--dir', type=str, default=None, help='Output directory')
    parser.add_argument('--perf_obj', default='MAX', help='Performance objective') # 'MAX', 'MIN', 'ZERO', 'ABS_MAX' or float value
    parser.add_argument('--popsize', type=int, default=96, help='Population size')    
    parser.add_argument('--max_gen', type=int, default=10, help='Number of generations')    

    # simulation arguments    
    parser.add_argument('--num_neurons', type=int, default=2, help='Number of neurons in agent')          
    parser.add_argument('--num_trials', type=int, default=4, help='Number of trials')        
    parser.add_argument('--trial_duration', type=int, default=50, help='Trial duration')        
    parser.add_argument('--num_random_pairings', type=int, default=None, help= \
        'None -> agents are alone in the simulation (default). '
        '0    -> agents are evolved in pairs: a genotype contains a pair of agents. '
        'n>0  -> each agent will go though a simulation with N other agents (randomly chosen).')        
    parser.add_argument('--mix_agents_motor_control', type=bool, default=False, help= \
        'when num_agents is 2 this decides whether the two agents switch control of L/R motors '
        'in different trials (mix=True) or not (mix=False) in which case the first agent '
        'always control the left motor and the second the right')
    parser.add_argument('--exclusive_motors_threshold', type=float, default=None, help = \
        'prevent motors to run at the same time')
    parser.add_argument('--dual_population', type=bool, default=False, help= \
        'If to evolve two separate populations, one always controlling the left '
        'motor and the other the right')        
    parser.add_argument('--cores', type=int, default=1, help='Number of cores')          

    args = parser.parse_args()

    t = TicToc()
    t.tic()

    genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE(args.num_neurons)
    genotype_size = gen_structure.get_genotype_size(genotype_structure)

    if args.dir is not None:
        # create default path if it specified dir already exists
        if os.path.isdir(args.dir):
            subdir = '{}n'.format(args.num_neurons)
            if args.num_random_pairings is not None:
                subdir += '_rp-{}'.format(args.num_random_pairings)
            if args.mix_agents_motor_control:
                subdir += '_mix'
            if args.exclusive_motors_threshold is not None:
                subdir += '_exc-{}'.format(args.exclusive_motors_threshold)
            if args.dual_population:
                subdir += '_dual'.format(args.exclusive_motors_threshold)
            seed_dir = 'seed_{}'.format(str(args.seed).zfill(3))
            outdir = os.path.join(args.dir,subdir,seed_dir)            
        else:
            # use the specified dir if it doesn't exist 
            outdir = args.dir
        utils.make_dir_if_not_exists_or_replace(outdir)
    else:
        outdir = None

    checkpoint_interval=int(np.ceil(args.max_gen/10))

    sim = Simulation(        
        genotype_structure = genotype_structure,        
        num_trials = args.num_trials,
        trial_duration = args.trial_duration,  # the brain would iterate trial_duration/brain_step_size number of time
        num_random_pairings = args.num_random_pairings,
        mix_agents_motor_control = args.mix_agents_motor_control,
        exclusive_motors_threshold = args.exclusive_motors_threshold,        
        dual_population = args.dual_population,
        num_cores = args.cores
    )

    if outdir is not None:      
        sim_config_json = os.path.join(outdir, 'simulation.json')  
        sim.save_to_file(sim_config_json)

    if args.num_random_pairings==0:
        genotype_size *= 2 # two agents per genotype
    
    evo = Evolution(
        random_seed=args.seed,
        num_populations= 2 if args.dual_population else 1,
        population_size=args.popsize,
        genotype_size=genotype_size, 
        evaluation_function=sim.evaluate,
        performance_objective=args.perf_obj,
        fitness_normalization_mode='FPS', # 'NONE', 'FPS', 'RANK', 'SIGMA' -> NO NORMALIZATION
        selection_mode='RWS', # 'UNIFORM', 'RWS', 'SUS'
        reproduce_from_elite=False,
        reproduction_mode='GENETIC_ALGORITHM',  # 'HILL_CLIMBING',  'GENETIC_ALGORITHM'
        mutation_variance=0.05, # mutation noice with variance 0.1
        elitist_fraction=0.05, # elite fraction of the top 4% solutions
        mating_fraction=0.95, # the remaining mating fraction (consider leaving something for random fill)
        crossover_probability=0.1,
        crossover_mode='UNIFORM',
        crossover_points= None, #genotype_structure['crossover_points'],
        folder_path=outdir,
        max_generation=args.max_gen,
        termination_function=None,
        checkpoint_interval=checkpoint_interval 
    )
    print('Output path: ', outdir)
    print('n_elite, n_mating, n_filling: ', evo.n_elite, evo.n_mating, evo.n_fillup)
    evo.run()

    print('Ellapsed time: {}'.format(t.tocvalue()))
