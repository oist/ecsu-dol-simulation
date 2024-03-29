"""
Rerun a simulation of a given seed and optionally visualize
animation and data plots of behavior and neural activity.
Run as
python -m dol.run_from_dir --help
"""

import os
from numpy.random import RandomState
from dol.simulation import Simulation
from pyevolver.evolution import Evolution
from dol import utils
import numpy as np
from dol.utils import get_numpy_signature


def run_simulation_from_dir(dir, generation=None, genotype_idx=0, population_idx=None,
                            random_target_seed=None, random_pairing_seed=None, 
                            isolation_idx=None, init_state=0., ghost_idx=None, 
                            write_data=False, **kwargs):
    """
    Utitity function to get data from a simulation
    """
    func_arguments = locals()

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

    if population_idx is None:
        # by default get the population with best performance
        population_idx = np.argmax(evo.best_performances[-1])
            

    # when referring to evo we use this
    population_idx_evo = population_idx 

    if sim.num_random_pairings == 0:
        # joined condition
        population_idx_evo = 0 # there is only 1 population for evo        

    data_record_list = []

    random_seed = evo.pop_eval_random_seed 
    # seed from evolution to reproduce results

    expect_same_results = True

    # overwriting simulaiton
    if random_target_seed is not None:
        print("Using random target")
        # standard target was initialized in sim.__post_init__
        # so this is going to overwrite it
        sim.init_target(RandomState(random_target_seed))
        expect_same_results = False
    if random_pairing_seed is not None:
        print("Setting random pairing with seed ", random_pairing_seed)
        random_seed = random_pairing_seed # new seed to get different results
        expect_same_results = False

    original_population = evo.population_unsorted

    # get the indexes of the populations as they were before being sorted by performance
    # we only need to do this for the first population (index 0)
    original_genotype_idx = evo.population_sorted_indexes[population_idx_evo][genotype_idx]

    played_back_data_record_list = None

    with_ghost = ghost_idx is not None

    if isolation_idx is not None:        
        assert sim.num_agents == 2, "sim already with single angent"
        assert not with_ghost, 'We cannot have both ghost and isolation mode'

        print('\n🏝️ Isolation mode - original simulation')

        # reset values for re-run
        func_arguments['isolation_idx'] = None
        func_arguments['write_data'] = None
        func_arguments['verbose'] = True


        # gather played back data
        _, sim_perfs, _, _, original_data_record_list, _ = \
            run_simulation_from_dir(**func_arguments)
        
        sim_idx = kwargs.get('select_sim') 
        if sim_idx is None: 
            sim_idx = np.argmin(sim_perfs) # best sim by default
        data_record = original_data_record_list[sim_idx]
        isolated_genotype = data_record['genotypes'][isolation_idx]

        isolated_agent_signature = get_numpy_signature(isolated_genotype)
        print(f'\n🏝️ New simulation with isolated agent ({isolated_agent_signature})')
        
        original_population = [[isolated_genotype]]
        population_idx = 0
        original_genotype_idx = 0       
        sim.agents = [sim.agents[0]] # take first, it will be reinitialized in run_simulation
        sim.num_agents = 1
        sim.num_random_pairings = None
        expect_same_results = False

    if with_ghost:
        print('\n👻 Ghost condition - original simulation')

        # reset values for re-run
        func_arguments['ghost_idx'] = None        
        func_arguments['random_target_seed']=None
        func_arguments['init_state'] = 0.
        func_arguments['write_data'] = None
        func_arguments['verbose'] = True
        func_arguments['compute_complexity'] = False
        
        # gather played back data
        _, _, _, _, played_back_data_record_list, _ = \
            run_simulation_from_dir(**func_arguments)        
        
        print('\n👻 New simulation with played back data')
        expect_same_results = False
        
    performance, sim_perfs, _ = sim.run_simulation(
        genotype_population=original_population,
        genotype_index=original_genotype_idx,
        random_seed=random_seed,
        population_index=population_idx,
        exaustive_pairs=True,
        init_ctrnn_state=init_state,
        data_record_list=data_record_list,
        ghost_index=ghost_idx,
        original_data_record_list=played_back_data_record_list
    )

    performance = sim.normalize_performance(performance)
    sim_perfs = [sim.normalize_performance(p) for p in sim_perfs]

    for sim_data in data_record_list:        
        sim_data['trials_performances'] = [
            sim.normalize_performance(p) for p in sim_data['trials_performances']
        ]
        sim_data['sim_performance'] = sim.normalize_performance(sim_data['sim_performance'])

    verbose = not kwargs.get('quiet', False)

    if verbose:
        selected_agent_genome = original_population[population_idx][original_genotype_idx]
        if sim.num_random_pairings == 0:
            # joined condition
            selected_agent_genome = np.split(selected_agent_genome, 2)[population_idx]    
        selected_agent_signature = get_numpy_signature(selected_agent_genome)        
        p_idx = population_idx if sim.num_random_pairings != 0 else ['left','right'][population_idx]
        print(f'Selected agent "{selected_agent_signature}" (pop_idx:{p_idx}, genome_idx:{genotype_idx})')
        perf_orig = evo.performances[population_idx_evo][genotype_idx]
        perf_orig = sim.normalize_performance(perf_orig)
        print("Error original: {}".format(perf_orig))
        print("Error recomputed: {}".format(performance))
        if expect_same_results:
            diff_perfomance = abs(perf_orig - performance)
            if diff_perfomance > 1e-5:
                print(f'Warning: diff_perfomance: {diff_perfomance}')
            # assert diff_perfomance < 1e-5, f'diff_perfomance: {diff_perfomance}'
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
                outfile = os.path.join(outdir, '{}.json'.format(k))
                utils.save_json_numpy_data(v, outfile)

    if kwargs.get('select_sim', None) is None:
        if with_ghost:
            # select worst one    
            sim_idx = np.argmax(sim_perfs)
            if verbose:
                print("Worst sim (random pairings)", sim_idx+1)
        else:
            # select best one
            sim_idx = np.argmin(sim_perfs)
            if verbose:
                if sim.num_agents == 1:
                    print("Single agent (single sim)", sim_idx+1)
                else:
                    print("Best sim (random pairings)", sim_idx+1)
    else:
        sim_idx = kwargs['select_sim'] - 1  # zero based
        if verbose:
            print("Selecting simulation", sim_idx+1)

    if verbose:
        from dol.analyze_results import get_non_flat_neuron_data
        data_record = data_record_list[sim_idx]
        print(f"   Error recomputed (sim {sim_idx+1}): ", sim_perfs[sim_idx])
        print("   Trials performances: ", data_record['trials_performances'])

        if sim.num_agents == 2:
            print("   Sim agents genotype distance: ", sim.agents_genotype_distance[sim_idx])
        # print agents signatures
        agents_sign = [get_numpy_signature(gt) for gt in data_record['genotypes']]
        print('   Agent(s) signature(s):', agents_sign)
        if sim.num_agents > 1:
            print('   Index of selected agent:', agents_sign.index(selected_agent_signature))
        non_flat_neurons = get_non_flat_neuron_data(data_record, 'agents_brain_output')
        print(f'   Non flat neurons: {non_flat_neurons}')


    if kwargs.get('compute_complexity', False):
        from dol.info_analysis.analyze_complexity import get_sim_agent_complexity
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
    parser.add_argument('--population_idx', type=int, default=None,
                        help='Index of the population, default pop with best agent fitness')
    parser.add_argument('--random_target_seed', type=int,
                        help='Seed to re-run simulation with random target (None to obtain same results)')
    parser.add_argument('--random_pairing_seed', type=int,
                        help='Seed to re-run simulation with random pairing (None to obtain same results)')
    parser.add_argument('--isolation_idx', type=int, choices=[0, 1], default=None,
                        help='To force the first (0) or second (1) agent to run in isolation (None otherwise) '
                             'Be aware that best agent could be at idx 1 (e.g., single population, rp-3)')
    parser.add_argument('--init_state', type=float, default=0.,
                        help="To force the agents' CTRNN to be initialized with specific initial state (default is 0.)")
    parser.add_argument('--ghost_idx', type=int, choices=[0, 1], default=None,
                        help='To force ghost condition specifying which agent is played back '
                             'Be aware that best agent could be at idx 1 (e.g., single population, rp-3)')
    parser.add_argument('--write_data', action='store_true', help='Whether to output data (same directory as input)')
    parser.add_argument('--select_sim', type=int, help='Which simulation to select for visualization and plot (1-based) - default best')
    parser.add_argument('--compute_complexity', action='store_true', help='Whether to plot the data')

    # additional args
    parser.add_argument('--visualize_trial', type=int, default=-1, help='Whether to visualize a certain trial (one-based)')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the data')
    parser.add_argument('--plot_trial', type=int, help='Whether to plot a specif trial (1-based)')

    args = parser.parse_args()

    perf, sim_perfs, evo, sim, data_record_list, sim_idx = \
        run_simulation_from_dir(**vars(args))

    data_record = data_record_list[sim_idx]

    if args.visualize_trial > 0:
        vis = Visualization(sim) if sim.num_dim == 1 else Visualization2D(sim)
        vis.start_simulation_from_data(args.visualize_trial - 1, data_record)
    if args.plot:
        plot.plot_results(evo, sim, args.plot_trial, data_record)

