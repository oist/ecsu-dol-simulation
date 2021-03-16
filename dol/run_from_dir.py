"""
Rerun a simulation of a given seed and optionally visualize
animation and data plots of behavior and neural activity.
Run as
python -m dol.run_from_dir --help
"""

import os
from numpy.random import RandomState
from dol.simulation import Simulation, MAX_MEAN_DISTANCE
from pyevolver.evolution import Evolution
from dol import utils
import numpy as np
from dol.utils import get_numpy_signature


def run_simulation_from_dir(dir, generation, genotype_idx=0, population_idx=0,
                            random_target_seed=None, random_pairing_seed=None, 
                            isolation_idx=None, write_data=False, **kwargs):
    """
    Utitity function to get data from a simulation
    """

    evo_files = sorted([f for f in os.listdir(dir) if f.startswith('evo_')])
    assert len(evo_files) > 0, "Can't find evo files in dir {}".format(dir)
    file_num_zfill = len(evo_files[0].split('_')[1].split('.')[0])
    if generation is None:
        evo_json_filepath = os.path.join(dir, evo_files[-1])
        generation = int(evo_files[-1].split('_')[1].split('.')[0])
    else:
        generation_str = str(generation).zfill(file_num_zfill)
        evo_json_filepath = os.path.join(dir, 'evo_{}.json'.format(generation_str))    
    sim_json_filepath = os.path.join(dir, 'simulation.json')    
    sim = Simulation.load_from_file(sim_json_filepath)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)

    data_record_list = []

    random_seed = evo.pop_eval_random_seed

    expect_same_results = isolation_idx is None

    # overwriting simulaiton
    if random_target_seed is not None:
        print("Using random target")
        # standard target was initialized in sim.__post_init__
        # so this is going to overwrite it
        sim.init_target(RandomState(random_target_seed))
        expect_same_results = False
    if random_pairing_seed is not None:
        print("Setting random pairing with seed ", random_pairing_seed)
        random_seed = random_pairing_seed
        expect_same_results = False

    original_populations = evo.population_unsorted

    # get the indexes of the populations as they were before being sorted by performance
    # we only need to do this for the first population (index 0)
    original_genotype_idx = evo.population_sorted_indexes[population_idx][genotype_idx]

    performance, sim_perfs, _ = sim.run_simulation(
        original_populations,
        original_genotype_idx,
        random_seed,
        population_idx,
        isolation_idx,
        data_record_list
    )

    performance = sim.normalize_performance(performance)

    verbose = not kwargs.get('quiet', False)

    if verbose:
        if genotype_idx == 0:
            perf_orig = evo.best_performances[generation][population_idx]
            perf_orig = sim.normalize_performance(perf_orig)
            print("Performace original: {}".format(perf_orig))
        print("Performace recomputed: {}".format(performance))
        if expect_same_results:
            assert abs(perf_orig - performance) < 1e-5
        # if performance == perf_orig:
        #     print("Exact!!")

    if write_data:
        for s, data_record in enumerate(data_record_list, 1):
            if len(data_record_list) > 1:
                outdir = os.path.join(dir, 'data', 'sim_{}'.format(s))
            else:
                outdir = os.path.join(dir, 'data')
            utils.make_dir_if_not_exists_or_replace(outdir)
            for k, v in data_record.items():
                if type(v) is dict:
                    # summary
                    outfile = os.path.join(outdir, '{}.json'.format(k))
                    utils.save_json_numpy_data(v, outfile)
                else:
                    outfile = os.path.join(outdir, '{}.json'.format(k))
                    utils.save_json_numpy_data(v, outfile)

    
    if kwargs.get('select_sim', None) is None:
        # select best one
        sim_idx = np.argmax(sim_perfs)
        # if sim.num_random_pairings != None and sim.num_random_pairings > 0:
        if verbose:
            print("Best sim (random pairings)", sim_idx+1)
    else:
        sim_idx = kwargs['select_sim'] - 1  # zero based
        if verbose:
            print("Selecting simulation", sim_idx+1)

    if verbose:
        sim_perf = sim.normalize_performance(sim_perfs[sim_idx])
        print("Performance recomputed (sim): ",  sim_idx+1, sim_perf)
        if sim.num_agents == 2:
            print("Sim genotype agents similarity: ", sim.agents_similarity[sim_idx])
        # print agents signatures
        agents_sign = [get_numpy_signature(gt) for gt in data_record_list[sim_idx]['genotypes']]
        print('Agent(s) signature(s):', agents_sign) 


    if kwargs.get('compute_complexity', False):
        from dol.analyze_complexity import get_sim_agent_complexity
        for a in range(sim.num_agents):
            nc = get_sim_agent_complexity(
                sim_perfs, sim, data_record_list,
                agent_index=a,
                sim_idx=sim_idx,
                analyze_sensors=True,
                analyze_brain=True,
                analyze_motors=False,
                combined_complexity=False,
                only_part_n1n2=True,
                rs=RandomState(1)
            )
            print('TSE', a+1, nc)

    return performance, sim_perfs, evo, sim, data_record_list, sim_idx


if __name__ == "__main__":
    import argparse
    from dol import plot
    from dol.visual import Visualization
    from dol.visual2d import Visualization2D

    parser = argparse.ArgumentParser(
        description='Rerun simulation'
    )

    # args for run_simulation_from_dir
    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--quiet', type=str, help='Do not print extra information (e.g., originale performance)')
    parser.add_argument('--generation', type=int, help='Number of generation to load')
    parser.add_argument('--genotype_idx', type=int, default=0, help='Index of agent in population to load')
    parser.add_argument('--population_idx', type=int, default=0,
                        help='Index of the population, usually 0 can be 1 in dual populations')
    parser.add_argument('--random_target_seed', type=int,
                        help='Seed to re-run simulation with random target (None to obtain same results)')
    parser.add_argument('--random_pairing_seed', type=int,
                        help='Seed to re-run simulation with random pairing (None to obtain same results)')
    parser.add_argument('--isolation_idx', type=int,
                        help='To force the first (0) or second (1) agent to run in isolation (None otherwise)')
    parser.add_argument('--write_data', action='store_true', help='Whether to output data (same directory as input)')
    parser.add_argument('--select_sim', type=int, help='Which simulation to select for visualization and plot (1-based) - default best')
    parser.add_argument('--compute_complexity', action='store_true', help='Whether to plot the data')

    # additional args
    parser.add_argument('--visualize_trial', type=int, default=-1, help='Whether to visualize a certain trial')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the data')
    parser.add_argument('--plot_trial', type=int, help='Whether to plot a specif trial')

    args = parser.parse_args()

    perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(**vars(args))

    data_record = data_record_list[sim_idx]

    if args.visualize_trial > 0:
        vis = Visualization(sim) if sim.num_dim == 1 else Visualization2D(sim)
        vis.start_simulation_from_data(args.visualize_trial - 1, data_record)
    if args.plot:
        plot.plot_results(evo, sim, args.plot_trial, data_record)

