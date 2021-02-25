import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dol.run_from_dir import run_simulation_from_dir    
from dol.shannon_entropy import get_shannon_entropy_2d, get_shannon_entropy_dd, get_shannon_entropy_dd_simplified
from dol.neural_complexity import compute_neural_complexity
import pandas as pd 

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


def get_seeds_generations_complexities(dir, analyze_sensors=True, 
    analyze_brain=True, analyze_motors=True,
    pop_index=0, only_last_generation=False, 
    filter_performance_threshold = None,
    use_brain_derivative=False): 

    print('dir', dir, 'pop_idx', pop_index)   

    seed_num_list = sorted([
        int(f.split('_')[1]) for f in os.listdir(dir)
        if f.startswith('seed_')
    ])

    GEN, NC, H = [], [], []
    skp_seeds = []

    for seed_num in tqdm(seed_num_list): 

        # print('seed', seed_num)

        seed_num_zero = str(seed_num).zfill(3)
        seed_dir = os.path.join(dir, 'seed_{}'.format(seed_num_zero))

        data_keys = []  # data elements on which complexity is analyzed
                        # trials x 1/2 agents x 500 data points x 2 dim                
        
        if analyze_sensors:
            data_keys.append('agents_sensors') # dim = num sensor = 2
        if analyze_brain:
            brain_key = 'agents_derivatives' if use_brain_derivative else 'agents_brain_output'
            data_keys.append(brain_key) # dim = num neurons = 2/4
        if analyze_motors:
            data_keys.append('agents_motors') # dim = num motors = 2

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
                a = pop_index # use the agent from the selected population
                trial_data_agent = data[:,t,a,:] 
                assert trial_data_agent.shape == (num_rows, num_data_points)                                
                nc = compute_neural_complexity(trial_data_agent) 
                # print("nc:",nc)
                nc_trials[t] = nc
                h_trials[t] = get_shannon_entropy_dd_simplified(np.transpose(trial_data_agent))
            nc_avg = np.mean(nc_trials)
            h_avg = np.mean(h_trials)
            # print("nc_avg:",nc_avg)
            # print("h_avg:",h_avg)
            nc_seed.append(nc_avg)
            h_seed.append(h_avg)      
            converged = filter_performance_threshold is None or perf < filter_performance_threshold
    
        if converged:
            GEN.append(num_generations_list)
            NC.append(nc_seed)
            H.append(h_seed)
        else:
            fill = [] if not only_last_generation else [np.NaN]
            GEN.append(fill)
            NC.append(fill)
            H.append(fill)
            skp_seeds.append(seed_num_zero)
    
    if len(skp_seeds)>0:
        print("Skipped seed", skp_seeds)

    return GEN, NC, H

def plot_generations_complexities(dir, analyze_sensors, 
    analyze_brain, analyze_motors, filter_performance_threshold):
    GEN, NC, H = get_seeds_generations_complexities(dir, analyze_sensors, 
        analyze_brain, analyze_motors, only_last_generation=False, 
        filter_performance_threshold=filter_performance_threshold)
    fig = plt.figure(figsize=(10, 6))
    num_plots = len(GEN)
    num_plot_cols = 5
    num_plot_rows = int(num_plots / num_plot_cols)
    if num_plots % num_plot_cols > 0:
        num_plot_cols += 1
    for seed_num, (num_gen_list, nc_seed, h_seed) in enumerate(zip(GEN, NC, H), 1):
        ax = fig.add_subplot(num_plot_rows, num_plot_cols, seed_num) 
        ax.plot(num_gen_list, nc_seed) # , label=str(seed_num)
        # ax.plot(num_generations_list, h_seed)
        # plt.legend()
    plt.show()

def main_line_plot():    
    analyze_sensors = True
    analyze_brain = True 
    analyze_motors = False
    dir = 'data/2n_exc-0.1/'
    plot_generations_complexities(
        dir, analyze_sensors, analyze_brain, analyze_motors,
        filter_performance_threshold=20.0)

def main_box_plot():
    num_neurons = 4
    analyze_sensors = False
    analyze_brain = True 
    analyze_motors = False     
    use_brain_derivatives = True   
    selected_nodes_str_list = [ 
        n
        for n,b in zip(
            ['sensors','dbrain' if use_brain_derivatives else 'brain','motors'],
            [analyze_sensors, analyze_brain, analyze_motors]
        ) if b
    ]
    all_NC = []
    for dir, pop_index in [
        (f'data/{num_neurons}n_exc-0.1/', 0),
        (f'data/{num_neurons}n_exc-0.1_rp-3_switch', 0), 
        (f'data/{num_neurons}n_exc-0.1_rp-3_dual', 0),
        (f'data/{num_neurons}n_exc-0.1_rp-3_dual', 1)]:                
        _, NC, _ = get_seeds_generations_complexities(
            dir, analyze_sensors, analyze_brain, analyze_motors, 
            pop_index, only_last_generation=True, filter_performance_threshold=20.0,
            use_brain_derivatives=use_brain_derivatives)
        
        NC = np.squeeze(NC)
        # print(NC)
        # print(NC.shape)
        all_NC.append(NC)

    all_NC = np.array(all_NC) # 4 x 20
    print(all_NC.shape)
    selected_nodes_file_str = '_'.join([x[:3] for x in selected_nodes_str_list])
    f_name = f"data/{num_neurons}n_{selected_nodes_file_str}.csv"
    print(f_name)
    # np.savetxt(f_name, all_NC, delimiter=",")
    x_labels = ['iso', 'gen', 'spec-left', 'spec-right']
    df = pd.DataFrame(np.transpose(all_NC), columns = x_labels) # 20 x 4
    df.to_csv(f_name)    
    all_NC_not_NaN = [x[~np.isnan(x)] for x in all_NC]
    plt.boxplot(all_NC_not_NaN, labels=x_labels)
    selected_nodes_str = ', '.join(selected_nodes_str_list)
    plt.title(f'Neural Complexity - {num_neurons}n ({selected_nodes_str})')
    plt.show()

# TODO: from a given seed, look at the last generation, 
# and compute the neural complexity for all agents
# plot correlation between fitness and complexity

if __name__ == "__main__":    
    # main_line_plot()
    main_box_plot()
    
    
    
    
    