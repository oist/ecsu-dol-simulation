"""
TODO: Missing module docstring
"""

import os
from dol import gen_structure
from dol.simulation import Simulation
from dol import utils
from pyevolver.evolution import Evolution
import numpy as np

import argparse
from pytictoc import TicToc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run the Division of Labor Simulation'
    )

    parser.add_argument('--seed', type=int, default=0, help='Random seed')     
    parser.add_argument('--dir', type=str, default=None, help='Output directory')
    parser.add_argument('--cores', type=int, default=4, help='Number of cores')        
    parser.add_argument('--num_neurons', type=int, default=2, help='Number of neurons in agent')    
    parser.add_argument('--popsize', type=int, default=96, help='Population size')    
    parser.add_argument('--num_gen', type=int, default=500, help='Number of generations')    
    parser.add_argument('--trial_duration', type=int, default=200, help='Trial duration')    
    parser.add_argument('--perf_obj', default='MAX', help='Performance objective') # 'MAX', 'MIN', 'ZERO', 'ABS_MAX' or float value

    args = parser.parse_args()

    t = TicToc()
    t.tic()

    genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE(args.num_neurons)
    genotype_size = gen_structure.get_genotype_size(genotype_structure)

    if args.dir is not None:
        # create default path if it specified dir already exists
        if os.path.isdir(args.dir):
            subdir = '{}n'.format(args.num_neurons)
            seed_dir = 'seed_{}'.format(str(args.seed).zfill(3))
            outdir = os.path.join(args.dir,subdir,seed_dir)            
        else:
            # use the specified dir if it doesn't exist 
            outdir = args.dir
        utils.make_dir_if_not_exists_or_replace(outdir)
    else:
        outdir = None

    sim = Simulation(
        genotype_structure = genotype_structure,        
        trial_duration = args.trial_duration,  # the brain would iterate trial_duration/brain_step_size number of time
        num_cores = args.cores     
    )

    if args.dir is not None:        
        sim_config_json = os.path.join(outdir, 'simulation.json')
        sim.save_to_file(sim_config_json)
    
    evo = Evolution(
        random_seed=args.seed,
        population_size=args.popsize,
        genotype_size=genotype_size, 
        evaluation_function=sim.evaluate,
        performance_objective=args.perf_obj,
        fitness_normalization_mode='NONE', # 'NONE', FPS', 'RANK', 'SIGMA' -> NO NORMALIZATION
        selection_mode='UNIFORM', # 'UNIFORM', 'RWS', 'SUS'
        reproduce_from_elite=True,
        reproduction_mode='GENETIC_ALGORITHM',  # 'HILL_CLIMBING',  'GENETIC_ALGORITHM'
        mutation_variance=0.1, # mutation noice with variance 0.1
        elitist_fraction=0.04, # elite fraction of the top 4% solutions
        mating_fraction=0.96, # the remaining mating fraction
        crossover_probability=0.1,
        crossover_mode='UNIFORM',
        crossover_points= None, #genotype_structure['crossover_points'],
        folder_path=outdir,
        max_generation=args.num_gen,
        termination_function=None,
        checkpoint_interval=np.ceil(args.num_gen/100),
    )
    evo.run()

    print('Ellapsed time: {}'.format(t.tocvalue()))
