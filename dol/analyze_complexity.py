import os
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from tqdm import tqdm
from dol.run_from_dir import run_simulation_from_dir    
from dol.shannon_entropy import get_shannon_entropy_2d, get_shannon_entropy_dd, get_shannon_entropy_dd_simplified
from dol.neural_complexity import compute_neural_complexity
import pandas as pd 
from pyevolver.evolution import Evolution

def get_test_data():
    num_sensors = 2
    num_motors = 2
    num_neurons = 4
    num_agents = 1
    num_data_points = 500
    num_trials = 4

    data_record = {
        'agents_sensors': [
            [
                [ 
                    ['t{}_a{}_p{}_sens{}'.format(t,a,p,s) for s in range(num_sensors)] 
                    for p in range(num_data_points)    
                ] 
                for a in range(num_agents)            
            ] 
            for t in range(num_trials)
        ],
        'agents_brain_output': [
            [
                [ 
                    ['t{}_a{}_p{}_brain{}'.format(t,a,p,n) for n in range(num_neurons)] 
                    for p in range(num_data_points)
                ] 
                for a in range(num_agents)            
            ] 
            for t in range(num_trials)
        ],
        'agents_motors': [
            [
                [ 
                    ['t{}_a{}_p{}_mot{}'.format(t,a,p,m) for m in range(num_motors)]                 
                    for p in range(num_data_points)
                ] 
                for a in range(num_agents)            
            ] 
            for t in range(num_trials)
        ],
    }
    return data_record


def get_sim_agent_complexity(sim_perfs, sim, data_record_list,
    analyze_sensors, analyze_brain, analyze_motors, use_brain_derivatives, 
    combined_complexity, rs):

    data_keys = []  # data elements on which complexity is analyzed
                    # trials x 1/2 agents x 500 data points x 2 dim                
    
    if analyze_sensors:
        data_keys.append('agents_sensors') # dim = num sensor = 2
    if analyze_brain:
        brain_key = 'agents_derivatives' if use_brain_derivatives else 'agents_brain_output'
        data_keys.append(brain_key) # dim = num neurons = 2/4
    if analyze_motors:
        data_keys.append('agents_motors') # dim = num motors = 2


    best_sim_idx = np.argmax(sim_perfs)
    
    num_trials = sim.num_trials
    num_agents = sim.num_agents
    num_data_points = sim.num_data_points
    data_record = data_record_list[best_sim_idx]   
    # data_record = get_test_data()            
    
    num_sensors = np.array(data_record['agents_sensors']).shape[-1] if analyze_sensors else 0
    num_neurons = np.array(data_record['agents_brain_output']).shape[-1] if analyze_brain else 0
    num_motors = np.array(data_record['agents_motors']).shape[-1] if analyze_motors else 0
    num_rows = num_sensors + num_neurons + num_motors        

    data = [ 
        np.moveaxis(np.array(data_record[k]), 3, 0)     # moving last dim (num_sensors/num_neurons/num_mot) first
        for k in data_keys                              # (num_trials, num_agents, num_data_points, num_neurons) -> 
    ]                                                   # (num_neurons, num_trials, num_agents, num_data_points)                                        
    
    if analyze_brain:
        assert sim.num_brain_neurons == num_neurons            

    data = np.stack([r for d in data for r in d ]) # stacking all rows together            
    assert data.shape == (
        num_rows, 
        num_trials, 
        num_agents,
        num_data_points,                
    )
    
    nc_trials = np.zeros(num_trials)
    h_trials = np.zeros(num_trials)
    for t in range(num_trials):
        # print("trial:",t+1)

        a = sim.population_index  # can be 1 in dual mode or in split if 
                                # current agent is in the second part of population
                
        if combined_complexity:
            assert num_agents == 2
            trial_data = np.concatenate(
                (
                    data[:,t,a,:],
                    data[:,t,1-a,:]
                )
            )
            assert trial_data.shape == (2*num_rows, num_data_points)
        else:
            trial_data = data[:,t,a,:] 
            assert trial_data.shape == (num_rows, num_data_points)                                

        nc = compute_neural_complexity(trial_data, rs) 
        # print("nc:",nc)
        nc_trials[t] = nc
        h_trials[t] = get_shannon_entropy_dd_simplified(np.transpose(trial_data))
    nc_avg = np.mean(nc_trials)
    h_avg = np.mean(h_trials)
    # print("nc_avg:",nc_avg)
    # print("h_avg:",h_avg)
    return nc_avg, h_avg


def get_seeds_generations_complexities(dir, analyze_sensors=True, 
    analyze_brain=True, analyze_motors=True,
    pop_index=0, only_last_generation=False, 
    filter_performance_threshold = None,
    use_brain_derivatives=False,
    combined_complexity=False, 
    rs=None): 

    print('dir', dir, 'pop_idx', pop_index)   

    seed_num_list = sorted([
        int(f.split('_')[1]) for f in os.listdir(dir)
        if f.startswith('seed_')
    ])

    GEN, NC, H, BP = [], [], [], []
    skp_seeds = []

    for seed_num in tqdm(seed_num_list): 

        # print('seed', seed_num)

        seed_num_zero = str(seed_num).zfill(3)
        seed_dir = os.path.join(dir, 'seed_{}'.format(seed_num_zero))

        num_generations_list = sorted([
            int(f.split('_')[1].split('.')[0]) 
            for f in os.listdir(seed_dir) if f.startswith('evo')
        ])

        if only_last_generation:
            num_generations_list = num_generations_list[-1:]

        nc_seed = []
        h_seed = []

        for generation in num_generations_list:
            
            # print("generation:",generation)
            perf, sim_perfs, evo, sim, data_record_list = run_simulation_from_dir(
                seed_dir, generation, population_idx=pop_index, quiet=True)

            nc_avg, h_avg = get_sim_agent_complexity(
                sim_perfs, sim, data_record_list,
                analyze_sensors, analyze_brain, analyze_motors, use_brain_derivatives,
                combined_complexity, rs
            )

            nc_seed.append(nc_avg)
            h_seed.append(h_avg) 

        # check if converged (only usefull for last generation)     
        converged = filter_performance_threshold is None or perf < filter_performance_threshold
        
        # get all best performances throught all generations from last evolution        

        if converged:
            GEN.append(num_generations_list)
            BP.append(sim.normalize_performance(np.array(evo.best_performances)))
            NC.append(nc_seed)
            H.append(h_seed)
        else:
            fill = [] if not only_last_generation else [np.NaN]
            GEN.append(fill)
            BP.append(fill)
            NC.append(fill)
            H.append(fill)
            skp_seeds.append(seed_num_zero)
    
    if len(skp_seeds)>0:
        print("Skipped seed", skp_seeds)

    return GEN, BP, NC, H


def main_line_plot():    

    dir = './data/1d_4n_exc-0.1_zfill'
    pop_index = 0    

    analyze_sensors = True
    analyze_brain = True 
    analyze_motors = False
    filter_performance_threshold = None #20.0
    combined_complexity = False
    
    rs = RandomState(1)    
    
    GEN, BP, NC, H = get_seeds_generations_complexities(dir, 
        analyze_sensors, analyze_brain, analyze_motors, 
        pop_index=pop_index, only_last_generation=False, 
        filter_performance_threshold=filter_performance_threshold, 
        combined_complexity=combined_complexity, rs=rs)

    fig = plt.figure(figsize=(10, 6))
    num_plots = len(GEN)
    num_plot_cols = 5
    num_plot_rows = int(num_plots / num_plot_cols)
    if num_plots % num_plot_cols > 0:
        num_plot_rows += 1

    for seed_num, (num_gen_list, best_perfs, nc_seed, h_seed) in enumerate(zip(GEN, BP, NC, H), 1):
        ax1 = fig.add_subplot(num_plot_rows, num_plot_cols, seed_num) 
        ax1.plot(num_gen_list, nc_seed) # , label=str(seed_num)
        ax1.set_ylim(0,30)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(best_perfs)), best_perfs, color='orange')
        ax2.set_ylim(0,100)
        # ax.plot(num_generations_list, h_seed)
        # plt.legend()
    plt.show()

def main_box_plot():
    num_dim = 1
    num_neurons = 4    
    analyze_sensors = True
    analyze_brain = True 
    analyze_motors = False     
    use_brain_derivatives = False  
    combined_complexity = True
    
    rs = RandomState(1) 
    
    selected_nodes_str_list = [ 
        n for n,b in zip(
            ['sensors','dbrain' if use_brain_derivatives else 'brain','motors'],
            [analyze_sensors, analyze_brain, analyze_motors]
        ) if b
    ]
    
    all_NC = []        

    if combined_complexity:
        dir_pop_index = [
            (f'data/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_switch', 0),
            (f'data/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_dual', 0),
        ]
        x_labels = ['gen', 'spec']        
    else:
        dir_pop_index = [
            (f'data/{num_dim}d_{num_neurons}n_exc-0.1_zfill/', 0),
            (f'data/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_switch', 0),
            (f'data/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_dual', 0),
            (f'data/{num_dim}d_{num_neurons}n_exc-0.1_zfill_rp-3_dual', 1)
        ]
        x_labels = ['iso', 'gen', 'spec-left', 'spec-right']

    for dir, pop_index in dir_pop_index:                
        _, _, NC, _ = get_seeds_generations_complexities(
            dir, analyze_sensors, analyze_brain, analyze_motors, 
            pop_index, only_last_generation=True, filter_performance_threshold=20.0,
            use_brain_derivatives=use_brain_derivatives, 
            combined_complexity=combined_complexity, rs=rs)
        
        NC = np.squeeze(NC)
        # print(NC)
        # print(NC.shape)
        all_NC.append(NC)

    all_NC = np.array(all_NC) # 4 x 20
    print(all_NC.shape)
    selected_nodes_file_str = '_'.join([x[:3] for x in selected_nodes_str_list])
    combined_str = '_combined' if combined_complexity else ''
    f_name = f"data/{num_neurons}n_{selected_nodes_file_str}{combined_str}.csv"
    print(f_name)    

    # save file to csv
    df = pd.DataFrame(np.transpose(all_NC), columns = x_labels) # 20 x 4
    df.to_csv(f_name)    
    
    all_NC_not_NaN = [x[~np.isnan(x)] for x in all_NC]
    plt.boxplot(all_NC_not_NaN, labels=x_labels)
    selected_nodes_str = ', '.join(selected_nodes_str_list)
    title = f'Neural Complexity - {num_neurons}n ({selected_nodes_str})'
    if combined_complexity:
        title += ' (combined)'
    plt.title(title)
    plt.show()


def main_scatter_plot():
    '''
    from a given seed, look at the last generation, 
    and compute the neural complexity for all agents
    plot correlation between fitness and complexity
    '''
    seed_dir = 'data/2n_exc-0.1_zfill/seed_001'
    generation = 5000
    pop_index = 0    

    analyze_sensors = True
    analyze_brain = True 
    analyze_motors = False
    use_brain_derivatives = False
    
    combined_complexity = False
    rs = RandomState(1)

    evo_file = sorted([f for f in os.listdir(seed_dir) if 'evo_' in f])[0]
    evo_json_filepath = os.path.join(seed_dir, evo_file)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=None)

    pop_size = len(evo.population[0])
    print('pop_size', pop_size)

    perf_data = np.zeros(pop_size)
    nc_data = np.zeros(pop_size)

    for genotype_idx in tqdm(range(pop_size)):

        perf, sim_perfs, evo, sim, data_record_list = run_simulation_from_dir(
            seed_dir, generation, genotype_idx, population_idx=pop_index, quiet=True)

        nc_avg, h_avg = get_sim_agent_complexity(
            sim_perfs, sim, data_record_list,
            analyze_sensors, analyze_brain, analyze_motors, use_brain_derivatives,
            combined_complexity, rs
        )

        perf_data[genotype_idx] = perf
        nc_data[genotype_idx] = nc_avg
        
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1) 
    ax.scatter(perf_data, nc_data)    
    plt.xlabel('performance')
    plt.ylabel('nc')
    plt.show()


def single_agent(init_value = 'random'):
    from dol import simulation    
    from dol import gen_structure
    from numpy.random import RandomState
    num_neurons = 4
    rs = RandomState(1)
    if init_value == 'random':
        run_result, sim, data_record_list = simulation.get_simulation_data_from_random_agent(
            gen_struct = gen_structure.DEFAULT_GEN_STRUCTURE(num_neurons),
            rs = rs
        )
    else:
        run_result, sim, data_record_list = simulation.get_simulation_data_from_filled_agent(
            gen_struct = gen_structure.DEFAULT_GEN_STRUCTURE(num_neurons),
            value = init_value,
            rs = rs
        )
    total_performance, sim_perfs, random_agent_indexes = run_result
    
    nc, h = get_sim_agent_complexity(
        sim_perfs, sim, data_record_list,
        analyze_sensors=True, 
        analyze_brain=True, 
        analyze_motors=False, 
        use_brain_derivatives=False,
        combined_complexity=False,
        rs = rs
    )

    print('nc', nc)
    print('h', h)

def single_paired_agents():
    from dol.simulation import Simulation
    import json
    seed_dir = 'data/2n_exc-0.1_zfill/seed_001'
    generation = 5000
    population_idx = 0

    rs = RandomState(1)

    sim_json_filepath = os.path.join(seed_dir, 'simulation.json')    
    evo_json_filepath = os.path.join(seed_dir, 'evo_{}.json'.format(generation))    
    
    sim = Simulation.load_from_file(
        sim_json_filepath,
        switch_agents_motor_control = True, # forcing switch
        num_random_pairings = 1 # forcing to play with one another
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
        best_two_agent_pop, 0, 0, 0, None,
        data_record_list
    )

    nc, h = get_sim_agent_complexity(
        sim_perfs, sim, data_record_list,
        analyze_sensors=True, 
        analyze_brain=True, 
        analyze_motors=False, 
        use_brain_derivatives=False,
        combined_complexity=False,
        rs = rs
    )

    print('performance', performance)
    print("Sim agents similarity: ", sim.agents_similarity[0])
    print('nc', nc)
    print('h', h)



if __name__ == "__main__":    
    # main_line_plot()
    # main_box_plot()
    # single_agent(0)
    single_paired_agents()
    # main_scatter_plot()
    
    
    
    
    