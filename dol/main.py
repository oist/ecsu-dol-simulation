"""
Runs evolutionary code for the main simulation.
Run from command line as
python -m dol.main args
See help for required arguments:
python -m dol.main --help
"""

import os
import argparse
from pytictoc import TicToc
import numpy as np
from pyevolver.evolution import Evolution
from dol import gen_structure
from dol import utils
from dol.simulation import Simulation


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description='Run the Division of Labor Simulation'
    )

    # evolution arguments
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dir', type=str, default=None, help='Output directory')
    parser.add_argument('--perf_obj', default='MAX',
                        help='Performance objective')  # 'MAX', 'MIN', 'ZERO', 'ABS_MAX' or float value
    parser.add_argument('--gen_zfill', action='store_true', default=False,
                        help='whether to fill geotipes with zeros otherwize random (default)')
    parser.add_argument('--num_pop', type=int, default=1, help='Number of populations')
    parser.add_argument('--popsize', type=int, default=96, help='Population size')
    parser.add_argument('--noshuffle', action='store_true', default=False, help='Weather to shuffle agents before eval function')
    parser.add_argument('--max_gen', type=int, default=10, help='Number of generations')

    # simulation arguments        
    parser.add_argument('--num_neurons', type=int, default=2, help='Number of neurons in agent')
    parser.add_argument('--num_dim', type=int, choices=[1, 2], default=1, help='Number of dimensions of the simulation')
    parser.add_argument('--num_trials', type=int, default=4, help='Number of trials')
    parser.add_argument('--trial_duration', type=int, default=50, help='Trial duration')
    parser.add_argument('--num_random_pairings', type=int, default=None,
                        help='None -> agents are alone in the simulation (default). '
                             '0    -> agents are evolved in pairs: a genotype contains a pair of agents. '
                             'n>0  -> each agent will go though a simulation with N other agents (randomly chosen).')
    parser.add_argument('--motor_control_mode', type=str, default=None,
                        choices=[None, 'SEPARATE', 'SWITCH', 'SWITCH-HM', 'OVERLAP'],
                        help=
                        'Type of motor control'
                        'None: not applicable (if single agent)'
                        'SEPARATE: across trials the first agent always control the left motor and the second the right'
                        'SWITCH: the two agents switch control of L/R motors in different trials'
                        'SWITCH-HM: the two agents switch control of L/R motors in different trials but only first half of motors are used'
                        'OVERLAP: both agents control L/R motors (for a factor o half)')

    parser.add_argument('--exclusive_motors_threshold', type=float, default=None,
                        help='prevent motors to run at the same time')    
    parser.add_argument('--cores', type=int, default=1, help='Number of cores')

    # Gather the provided arguements as an array.
    args = parser.parse_args(raw_args)

    t = TicToc()
    t.tic()

    genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE(args.num_dim, args.num_neurons)
    # agents in 2d have 4 sensors and 4 motors
        
    genotype_size = gen_structure.get_genotype_size(genotype_structure)

    if args.dir is not None:
        # create default path if it specified dir already exists
        if os.path.isdir(args.dir):
            subdir = f'{args.num_dim}d_{args.num_neurons}n'
            if args.exclusive_motors_threshold is not None:
                subdir += '_exc-{}'.format(args.exclusive_motors_threshold)
            if args.gen_zfill:
                subdir += '_zfill'
            if args.num_random_pairings is not None:
                subdir += '_rp-{}'.format(args.num_random_pairings)
            if args.num_pop > 1:
                subdir += f'_np-{args.num_pop}'
            if args.noshuffle:
                subdir += f'_noshuffle'
            if args.motor_control_mode!=None:
                subdir += f'_{args.motor_control_mode.lower()}'
            seed_dir = 'seed_{}'.format(str(args.seed).zfill(3))
            outdir = os.path.join(args.dir, subdir, seed_dir)
        else:
            # use the specified dir if it doesn't exist 
            outdir = args.dir
        utils.make_dir_if_not_exists_or_replace(outdir)
    else:
        outdir = None

    checkpoint_interval = int(np.ceil(args.max_gen / 10))

    sim = Simulation(
        genotype_structure=genotype_structure,
        num_pop=args.num_pop,
        num_dim=args.num_dim,
        num_trials=args.num_trials,
        trial_duration=args.trial_duration,  # the brain would iterate trial_duration/brain_step_size number of time
        num_random_pairings=args.num_random_pairings,
        motor_control_mode=args.motor_control_mode,
        exclusive_motors_threshold=args.exclusive_motors_threshold,
        num_cores=args.cores
    )

    if outdir is not None:
        sim_config_json = os.path.join(outdir, 'simulation.json')
        sim.save_to_file(sim_config_json)

    if args.num_random_pairings == 0:
        genotype_size *= 2  # two agents per genotype

    population = None  # by default randomly initialized in evolution

    if args.gen_zfill:
        # all genotypes initialized with zeros
        population = np.zeros(
            (args.num_pop, args.popsize, genotype_size)
        )

    evo = Evolution(
        random_seed=args.seed,
        population=population,
        num_populations=args.num_pop,
        shuffle_agents=not args.noshuffle,
        population_size=args.popsize,
        genotype_size=genotype_size,
        evaluation_function=sim.evaluate,
        performance_objective=args.perf_obj,
        fitness_normalization_mode='FPS',  # 'NONE', 'FPS', 'RANK', 'SIGMA' -> NO NORMALIZATION
        selection_mode='RWS',  # 'UNIFORM', 'RWS', 'SUS'
        reproduce_from_elite=False,
        reproduction_mode='GENETIC_ALGORITHM',  # 'HILL_CLIMBING',  'GENETIC_ALGORITHM'
        mutation_variance=0.05,  # mutation noice with variance 0.1
        elitist_fraction=0.05,  # elite fraction of the top 4% solutions
        mating_fraction=0.95,  # the remaining mating fraction (consider leaving something for random fill)
        crossover_probability=0.1,
        crossover_mode='UNIFORM',
        crossover_points=None,  # genotype_structure['crossover_points'],
        folder_path=outdir,
        max_generation=args.max_gen,
        termination_function=None,
        checkpoint_interval=checkpoint_interval
    )
    print('Output path: ', outdir)
    print('n_elite, n_mating, n_filling: ', evo.n_elite, evo.n_mating, evo.n_fillup)
    evo.run()

    print('Ellapsed time: {}'.format(t.tocvalue()))

    return sim, evo


if __name__ == "__main__":
    main()
