"""
Main script to analyze neural complexity.
Measures currently available: Shannon Entropy and TSE complexity.
"""
import os
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from tqdm import tqdm
from dol.run_from_dir import run_simulation_from_dir
from dol.shannon_entropy import get_shannon_entropy_2d, get_shannon_entropy_dd, get_shannon_entropy_dd_simplified
from dol.neural_complexity import compute_neural_complexity, compute_neural_complexity_n1n2_combined, compute_neural_complexity_n1n2_single
import pandas as pd
from pyevolver.evolution import Evolution


def get_test_data():
    num_sensors_motors = 2
    num_neurons = 4
    num_agents = 1
    num_data_points = 500
    num_trials = 4

    data_record = {
        'agents_sensors': [
            [
                [
                    ['t{}_a{}_p{}_sens{}'.format(t, a, p, s) for s in range(num_sensors_motors)]
                    for p in range(num_data_points)
                ]
                for a in range(num_agents)
            ]
            for t in range(num_trials)
        ],
        'agents_brain_output': [
            [
                [
                    ['t{}_a{}_p{}_brain{}'.format(t, a, p, n) for n in range(num_neurons)]
                    for p in range(num_data_points)
                ]
                for a in range(num_agents)
            ]
            for t in range(num_trials)
        ],
        'agents_motors': [
            [
                [
                    ['t{}_a{}_p{}_mot{}'.format(t, a, p, m) for m in range(num_sensors_motors)]
                    for p in range(num_data_points)
                ]
                for a in range(num_agents)
            ]
            for t in range(num_trials)
        ],
    }
    return data_record


def get_sim_agent_complexity(sim_perfs, sim, data_record_list, agent_index, sim_idx,
                             analyze_sensors, analyze_brain, analyze_motors,
                             combined_complexity, only_part_n1n2, rs):
    data_keys = []  # data elements on which complexity is analyzed
    # trials x 1/2 agents x 500 data points x 2 dim

    if analyze_sensors:
        data_keys.append('agents_sensors')  # dim = num sensor = 2
    if analyze_brain:
        brain_key = 'agents_brain_output'
        data_keys.append(brain_key)  # dim = num neurons = 2/4
    if analyze_motors:
        data_keys.append('agents_motors')  # dim = num motors = 2

    if sim_idx is None:
        sim_idx = np.argmin(sim_perfs)

    num_trials = sim.num_trials
    num_agents = sim.num_agents
    num_data_points = sim.num_data_points
    data_record = data_record_list[sim_idx]
    # data_record = get_test_data()            

    num_sensors = np.array(data_record['agents_sensors']).shape[-1] if analyze_sensors else 0
    num_neurons = np.array(data_record['agents_brain_output']).shape[-1] if analyze_brain else 0
    num_motors = np.array(data_record['agents_motors']).shape[-1] if analyze_motors else 0
    num_rows = num_sensors + num_neurons + num_motors

    data = [ 
        np.moveaxis(np.array(data_record[k]), 3, 0)     # moving last dim (num_sensors_motors/num_neurons/num_mot) first
        for k in data_keys                              # (num_trials, num_agents, num_data_points, num_neurons) -> 
    ]                                                   # (num_neurons, num_trials, num_agents, num_data_points)                                        
    
    if analyze_brain:
        assert sim.num_brain_neurons == num_neurons

    data = np.stack([r for d in data for r in d])  # stacking all rows together
    assert data.shape == (
        num_rows,
        num_trials,
        num_agents,
        num_data_points,
    )

    nc_trials = np.zeros(num_trials)
    for t in range(num_trials):
        # print("trial:",t+1)

        if agent_index is None:
            # we get the agent_index based on what is the best agent
            # can be 1 in dual mode if manually set 
            # or in split-mode if best agent was in the second part of population 
            # (when it was shuffled)
            a = sim.population_index
        else:
            a = agent_index          

        if only_part_n1n2:
            n1_idx = num_sensors
            n2_idx = num_sensors+1
            if combined_complexity:
                trial_data1 = data[:, t, a, :]
                trial_data2 = data[:, t, 1 - a, :]                
                nc = compute_neural_complexity_n1n2_combined(
                    trial_data1, trial_data2, n1_idx, n2_idx, rs)
            else:
                trial_data = data[:, t, a, :]
                nc = compute_neural_complexity_n1n2_single(
                    trial_data, n1_idx, n2_idx, rs)
        else:
            if combined_complexity:
                assert num_agents == 2
                trial_data = np.concatenate(
                    (
                        data[:, t, a, :],
                        data[:, t, 1 - a, :]
                    )
                )
                assert trial_data.shape == (2 * num_rows, num_data_points)
            else:
                trial_data = data[:, t, a, :]
                assert trial_data.shape == (num_rows, num_data_points)

            nc = compute_neural_complexity(trial_data, rs)

        # print("nc:",nc)
        nc_trials[t] = nc
    nc_avg = np.mean(nc_trials)
    # print("nc_avg:",nc_avg)
    # print("h_avg:",h_avg)
    return nc_avg


def get_seeds_generations_complexities(
        in_dir, analyze_sensors=True, analyze_brain=True, analyze_motors=True,
        pop_index=0, only_last_generation=False, filter_performance_threshold=None,
        combined_complexity=False,
        only_part_n1n2=False, rs=None):

    print('in_dir', in_dir, 'pop_idx', pop_index)

    seeds = sorted([
        int(f.split('_')[1]) for f in os.listdir(in_dir)
        if f.startswith('seed_')
    ])

    SEEDS, GEN, NC, BP = [], [], [], []
    skp_seeds = []

    for seed_num in tqdm(seeds):

        # print('seed', seed_num)

        seed_num_zero = str(seed_num).zfill(3)
        seed_dir = os.path.join(in_dir, 'seed_{}'.format(seed_num_zero))

        num_generations_list = sorted([
            int(f.split('_')[1].split('.')[0])
            for f in os.listdir(seed_dir) if f.startswith('evo')
        ])

        if only_last_generation:
            num_generations_list = num_generations_list[-1:]

        nc_seed = []

        for generation in num_generations_list:
            # print("generation:",generation)
            perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(
                seed_dir, generation, population_idx=pop_index, quiet=True)

            agent_index = None # agent_index must be None (to get best agents of the two)
            sim_idx = None # sim_idx must be None (to get best sim among randomom pairs)

            nc_avg = get_sim_agent_complexity(
                sim_perfs, sim, data_record_list, agent_index, sim_idx,
                analyze_sensors, analyze_brain, analyze_motors,
                combined_complexity, only_part_n1n2, rs
            )

            nc_seed.append(nc_avg)

            # check if converged (only usefull for last generation)
        converged = filter_performance_threshold is None or perf < filter_performance_threshold

        # get all best performances throught all generations from last evolution        

        if converged:
            SEEDS.append(seed_num)
            GEN.append(num_generations_list)
            BP.append(sim.normalize_performance(np.array(evo.best_performances)))
            NC.append(nc_seed)  
        elif only_last_generation:
            fill = [np.NaN]
            GEN.append(fill)
            BP.append(fill)
            NC.append(fill)
            skp_seeds.append(seed_num_zero)
        # otherwise do not include them (uncomment if you want to include as NaN)
        # else:                        
        #     fill = np.full(len(num_generations_list), np.NaN)
        #     GEN.append(fill)
        #     BP.append(fill)
        #     NC.append(fill)
        #     skp_seeds.append(seed_num_zero)

    if len(skp_seeds) > 0:
        print("Skipped seed", skp_seeds)

    return SEEDS, GEN, BP, NC


def main_line_plot(num_dim = 1, num_neurons = 2, sim_type = 'individuals',
    analyze_sensors = True, analyze_brain = True, analyze_motors = False,
    tse_max=2, combined_complexity = False, only_part_n1n2 = True, 
    input_dir = 'data', plot_dir = None, csv_dir = None):
    """
    Given a hard-coded directory with N simulation seeds,
    plots N line subplots, each containing 2 lines:
    - performance over generations
    - hardcoded complexity over generations
    """    

    sim_type_dir = {
        'individuals': f'{num_dim}d_{num_neurons}n_exc-0.1_zfill',
        'generalists': f'{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_switch',
        'specialists': f'{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_dual'
    }
    
    assert sim_type in sim_type_dir, f'sim_type must be in {sim_type_dir.keys()}'
    
    exp_dir = sim_type_dir[sim_type]
    in_dir = os.path.join(input_dir, exp_dir)

    pop_index = 0

    rs = RandomState(1)

    SEEDS, GEN, BP, NC = get_seeds_generations_complexities(
        in_dir, analyze_sensors, analyze_brain, analyze_motors,
        pop_index=pop_index, only_last_generation=False,        
        filter_performance_threshold=20, # exclude not converged seeds
        combined_complexity=combined_complexity, 
        only_part_n1n2=only_part_n1n2,
        rs=rs
    )

    fig = plt.figure(figsize=(10, 6))
    num_plots = len(GEN)
    num_plot_cols = 5
    num_plot_rows = int(num_plots / num_plot_cols)
    if num_plots % num_plot_cols > 0:
        num_plot_rows += 1

    if csv_dir is not None or plot_dir is not None:
        selected_nodes_str_list = [
                n for n, b in zip(
                    ['sensors', 'brain', 'motors'],
                    [analyze_sensors, analyze_brain, analyze_motors]
                ) if b
            ]
        selected_nodes_file_str = '_'.join([x[:3] for x in selected_nodes_str_list])
        combined_str = '_combined' if combined_complexity else ''    
        only_part_n1n2_str = '_onlyN1N2' if only_part_n1n2 else ''    
        fname = f'{num_dim}d_{num_neurons}n_gen_seeds_TSE_{sim_type}_{selected_nodes_file_str}{combined_str}{only_part_n1n2_str}'

    if csv_dir is not None:
        # save file to csv    
    
        f_path = os.path.join(
            csv_dir, 
            fname + '.csv'
        )
        print('saving csv:', f_path)    

        num_seeds, gen_list_size = np.array(NC).shape
        assert gen_list_size == len(GEN[0])
        csv_data = np.transpose( # gen_list_size x num_seeds
            np.concatenate(
                [
                    np.array(GEN[0]).reshape((1,gen_list_size)), # need only 1 list of generations
                    NC
                ]
            )
        )
        seeds_str = [f'seed_{s}' for s in SEEDS]
        assert len(seeds_str) == num_seeds
        cols_names = ['GEN'] + seeds_str
        df = pd.DataFrame(csv_data, columns=cols_names) 
        df.to_csv(f_path, index=False)

    for seed_num, (num_gen_list, best_perfs, nc_seed) in enumerate(zip(GEN, BP, NC), 1):
        ax1 = fig.add_subplot(num_plot_rows, num_plot_cols, seed_num)
        ax1.plot(num_gen_list, nc_seed)  # , label=str(seed_num)
        ax1.set_ylim(0, tse_max)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(best_perfs)), best_perfs, color='orange')
        ax2.set_ylim(0, 100)
        # plt.legend()
    fig.tight_layout()
    
    selected_nodes_str = ', '.join(selected_nodes_str_list)    
    title = f'Perf & TSE on generations - {num_neurons}n {sim_type} {selected_nodes_str}'
    if combined_complexity:
        title += ' - combined'
    if only_part_n1n2:
        title += ' - onlyN1N2'
    
    fig.suptitle(title)    
    fig.subplots_adjust(top=0.88)

    if plot_dir is not None:
        f_path = os.path.join(
            plot_dir, 
            fname + '.pdf'
        )
        plt.savefig(f_path)  # remove csv and add pdf
        print('saving pdf:', f_path)        
    else:
        plt.show()


def main_box_plot(num_dim = 1, num_neurons = 2, analyze_sensors = True,
    analyze_brain = True, analyze_motors = False,
    combined_complexity = True, only_part_n1n2 = True, 
    input_dir = 'data', csv_dir = None, plot_dir = None):
    """
    Calculates neural complexity for different conditions,
    saves it to CSV and outputs boxplots.
    """
    

    rs = RandomState(1)

    selected_nodes_str_list = [
        n for n, b in zip(
            ['sensors', 'brain', 'motors'],
            [analyze_sensors, analyze_brain, analyze_motors]
        ) if b
    ]

    all_NC = []

    if combined_complexity:
        dir_pop_index = [
            (f'{input_dir}/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_switch', 0),
            (f'{input_dir}/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_dual', 0),
        ]
        x_labels = ['gen', 'spec']
    else:
        dir_pop_index = [
            (f'{input_dir}/{num_dim}d_{num_neurons}n_exc-0.1_zfill/', 0),
            (f'{input_dir}/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_switch', 0),
            (f'{input_dir}/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_dual', 0),
            (f'{input_dir}/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_dual', 1)
        ]
        x_labels = ['iso', 'gen', 'spec-left', 'spec-right']

    for in_dir, pop_index in dir_pop_index:
        SEEDS, GEN, BP, NC = get_seeds_generations_complexities(
            in_dir, analyze_sensors, analyze_brain, analyze_motors,
            pop_index, only_last_generation=True, filter_performance_threshold=20.0,            
            combined_complexity=combined_complexity, 
            only_part_n1n2=only_part_n1n2,
            rs=rs)

        NC = np.squeeze(NC)
        # print(NC)
        # print(NC.shape)
        all_NC.append(NC)

    all_NC = np.array(all_NC)  # 4 x 20
    print(all_NC.shape)
    selected_nodes_file_str = '_'.join([x[:3] for x in selected_nodes_str_list])
    combined_str = '_combined' if combined_complexity else ''    
    only_part_n1n2_str = '_onlyN1N2' if only_part_n1n2 else ''    

    f_name = f"{num_dim}d_{num_neurons}n_box_TSE_{selected_nodes_file_str}{combined_str}{only_part_n1n2_str}" 

    # save file to csv    
    if csv_dir is not None:
        f_path = os.path.join(
            csv_dir,
            f_name + '.csv'
        )
        print('saving csv:', f_path)
        df = pd.DataFrame(np.transpose(all_NC), columns=x_labels)  # 20 x 4
        df.to_csv(f_path, index=False)

    all_NC_not_NaN = [x[~np.isnan(x)] for x in all_NC]
    plt.boxplot(all_NC_not_NaN, labels=x_labels)
    selected_nodes_str = ', '.join(selected_nodes_str_list)
    title = f'Neural Complexity - {num_neurons}n {selected_nodes_str}'
    if combined_complexity:
        title += ' - combined'
    if only_part_n1n2:
        title += ' - onlyN1N2'
    plt.title(title)

    if plot_dir is not None:
        f_path = os.path.join(
            plot_dir,
            f_name + '.pdf'
        )
        plt.savefig(f_path)  # remove csv and add pdf
        print('saving pdf:', f_path)
        plt.clf()
    else:
        plt.show()


def main_scatter_plot(input_dir='data'):
    """
    From a given seed, look at the last generation,
    and compute the neural complexity for all agents.
    Plot correlation between fitness and complexity.
    """
    seed_dir = f'{input_dir}/2n_exc-0.1_zfill/seed_001'
    generation = 5000
    pop_index = 0

    analyze_sensors = True
    analyze_brain = True
    analyze_motors = False

    combined_complexity = False
    only_part_n1n2 = True

    rs = RandomState(1)

    evo_file = sorted([f for f in os.listdir(seed_dir) if 'evo_' in f])[0]
    evo_json_filepath = os.path.join(seed_dir, evo_file)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=None)

    pop_size = len(evo.population[0])
    print('pop_size', pop_size)

    perf_data = np.zeros(pop_size)
    nc_data = np.zeros(pop_size)

    for genotype_idx in tqdm(range(pop_size)):
        perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(
            seed_dir, generation, genotype_idx, population_idx=pop_index, quiet=True)

        agent_index = None # agent_index must be None (to get best agents of the two)
        sim_idx = None # sim_idx must be None (to get best sim among randomom pairs)

        nc_avg = get_sim_agent_complexity(
            sim_perfs, sim, data_record_list, agent_index, sim_idx,
            analyze_sensors, analyze_brain, analyze_motors,
            combined_complexity, only_part_n1n2, rs
        )

        perf_data[genotype_idx] = perf
        nc_data[genotype_idx] = nc_avg

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(perf_data, nc_data)
    plt.xlabel('performance')
    plt.ylabel('nc')
    plt.show()


def single_agent(init_value='random'):
    """
    Test neural complexity for a single agent initialized with some genotype.
    """
    from dol import simulation
    from dol import gen_structure
    from numpy.random import RandomState
    num_dim = 1
    num_neurons = 4
    rs = RandomState(1)
    if init_value == 'random':
        run_result, sim, data_record_list = simulation.get_simulation_data_from_random_agent(
            gen_struct=gen_structure.DEFAULT_GEN_STRUCTURE(num_dim, num_neurons),
            rs=rs
        )
    else:
        run_result, sim, data_record_list = simulation.get_simulation_data_from_filled_agent(
            gen_struct=gen_structure.DEFAULT_GEN_STRUCTURE(num_dim, num_neurons),
            value=init_value,
            rs=rs
        )
    total_performance, sim_perfs, paired_agents_sims_pop_idx = run_result

    nc = get_sim_agent_complexity(
        sim_perfs, sim, data_record_list,
        agent_index=None,
        sim_idx=None,
        analyze_sensors=True,
        analyze_brain=True,
        analyze_motors=False,
        combined_complexity=False,
        only_part_n1n2=False,
        rs=rs
    )

    print('nc', nc)


def single_paired_agents(input_dir='data'):
    """
    Test whether individually evolved agents can perform the task together
    and calculate their combined neural complexity.
    """
    from dol.simulation import Simulation
    import json
    seed_dir = f'{input_dir}/2n_exc-0.1_zfill/seed_001'
    generation = 5000
    population_idx = 0

    rs = RandomState(1)

    sim_json_filepath = os.path.join(seed_dir, 'simulation.json')
    evo_json_filepath = os.path.join(seed_dir, 'evo_{}.json'.format(generation))

    sim = Simulation.load_from_file(
        sim_json_filepath,
        motor_control_mode='SWITCH',  # forcing switch
        num_random_pairings=1  # forcing to play with one another
    )

    evo = Evolution.load_from_file(evo_json_filepath, folder_path=None)

    original_populations = evo.population_unsorted

    best_two_agent_pop = np.array([
        [
            original_populations[0][x] for x in
            evo.population_sorted_indexes[population_idx][:2]
        ]
    ])

    data_record_list = []

    performance, sim_perfs, _ = sim.run_simulation(
        genotype_population=best_two_agent_pop,
        genotype_index=0,
        random_seed=0,
        population_index=0,
        exaustive_pairs=True
    )

    nc = get_sim_agent_complexity(
        sim_perfs, sim, data_record_list,
        agent_index=None,
        sim_idx=None,
        analyze_sensors=True,
        analyze_brain=True,
        analyze_motors=False,
        combined_complexity=False,
        only_part_n1n2=False,
        rs=rs
    )

    print('performance', performance)
    print("Sim agents similarity: ", sim.agents_genotype_distance[0])
    print('nc', nc)

