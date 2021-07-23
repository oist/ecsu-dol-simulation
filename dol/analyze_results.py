"""
Given a directory with N simulation seeds,
returns a bar plot with best performance of last generation
of all seeds.
Run from command line as
python -m dol.analyze_results ./data/exp_dir
where 'exp_dir' contains all the simulation seeds
"""
import os
import json
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from dol.simulation import Simulation
from dol.run_from_dir import run_simulation_from_dir

CONVERGENCE_THRESHOLD = 20.
VARIANCE_THRESHOLD = 1e-6

def plot_best_exp_performance(best_exp_performance, seeds):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Convergence seeds')
    ind = np.arange(len(seeds))        
    num_bars = len(best_exp_performance[0])
    width = 0.7 / num_bars
    for p in range(num_bars):
        p_series = [b[p] for b in best_exp_performance]
        x_pos = ind + p * width + width/2
        ax.bar(x_pos, p_series, width, label=f'Pop{p+1}')
    ax.set_xticks(ind + 0.7 / 2)
    ax.set_xticklabels(seeds)
    plt.xlabel('Seeds')
    plt.ylabel('Error')    
    plt.legend(bbox_to_anchor=(-0.15, 1.10), loc='upper left')
    plt.show()

def bar_plot_seeds_data_list(seed_data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title)
    seeds = seed_data.keys()
    ind = np.arange(len(seeds))        
    num_bars = len(list(seed_data.values())[0])
    width = 0.7 / num_bars
    for p in range(num_bars):
        p_series = [b[p] for b in seed_data.values()]
        x_pos = ind + p * width + width/2
        ax.bar(x_pos, p_series, width, label=f'A{p+1}')
    ax.set_xticks(ind + 0.7 / 2)
    ax.set_xticklabels(seeds)
    plt.xlabel('Seeds')
    plt.ylabel('Non Flat Elements')
    plt.legend(bbox_to_anchor=(-0.15, 1.10), loc='upper left')
    plt.show()    

def bar_plot_seeds_data_value(seed_data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title)
    seeds = seed_data.keys()
    ind = np.arange(len(seeds))        
    width = 0.7
    p_series = list(seed_data.values())
    x_pos = ind + width/2
    ax.bar(x_pos, p_series, width)
    ax.set_xticks(ind + 0.7 / 2)
    ax.set_xticklabels(seeds)
    plt.xlabel('Seeds')
    plt.show()    

def get_non_flat_neuron_data(data_record, key):
    brain_data = data_record[key] # shape: (num_trials, num_agents, sim_steps(500), num_dim (num_neurons))
    brain_data = np.moveaxis(brain_data, (0,2), (2,3)) # (num_agents, num_dim (num_neurons), num_trials, sim_steps(500))
    brain_data = brain_data[:,:,:,100:] # cut the firs 100 point in each trial (brain outputs needs few steps to converge)        
    var = np.var(brain_data, axis=3) 
    max_var = np.max(var, axis=2) # for each agent, each neuron what is the max variance across trials  
    non_flat_neurons = np.sum(max_var > VARIANCE_THRESHOLD, axis=1)    
    return non_flat_neurons

def min_max_flat_elements(values):
    # values is a list of pairs [x,y] (e.g., [1,2])
    # indicating how many non-flat elements in the corresponding seed
    sorted_set = sorted(
        set([tuple(sorted(x)) for x in values]),
        key = lambda x: np.sum(x)
    )
    if len(sorted_set)==1:
        return str(list(sorted_set[0]))
    return f'{list(sorted_set[0])} ... {list(sorted_set[-1])}'

def get_last_performance_seeds(base_dir, print_stats=True, 
    print_values=False, plot=False, export_to_csv=False,
    best_sim_stats=None):

    exp_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('seed_')])
    best_exp_performance = []  # the list of best performances of last generation for all seeds
    all_gen_best_performances = [] # all best performances for all seeds for all generations
    last_evo_file = None
    seeds = []
    seed_exp_dir = {}
    for exp in exp_dirs:
        exp_dir = os.path.join(base_dir, exp)
        if last_evo_file is None:
            evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
            if len(evo_files) == 0:
                # no evo files
                continue
            last_evo_file = evo_files[-1]
        evo_file = os.path.join(exp_dir, last_evo_file)
        if not os.path.isfile(evo_file):
            continue
        with open(evo_file) as f_in:
            sim_json_filepath = os.path.join(exp_dir, 'simulation.json')    
            sim = Simulation.load_from_file(sim_json_filepath)
            exp_evo_data = json.load(f_in)
            s = exp_evo_data['random_seed']
            if s>20:
                continue
            seeds.append(s)
            seed_exp_dir[s] = exp_dir
            gen_best_perf = np.array(exp_evo_data['best_performances']) # one per population            

            # make sure it's monotonic increasing(otherwise there is a bug)
            # assert all(gen_best_perf[i] <= gen_best_perf[i+1] for i in range(len(gen_best_perf)-1))

            perf_index = lambda a: '|'.join(['{:.5f}'.format(x) for x in a])

            last_best_performance = gen_best_perf[-1] 
            last_best_performance = sim.normalize_performance(last_best_performance)           
            if print_values:
                print('{} {}'.format(exp, perf_index(last_best_performance)))            
            best_exp_performance.append(last_best_performance)
            all_gen_best_performances.append(gen_best_perf)

    converged_seeds = [s for s,p in zip(seeds,best_exp_performance) if np.min(p)<CONVERGENCE_THRESHOLD]    
    non_converged_seeds = [s for s in seeds if s not in converged_seeds]
    conv_seeds_err = {s:round(np.min(p),0) for s,p in zip(seeds,best_exp_performance) if s in converged_seeds}
    non_conv_seeds_err = {s:round(np.min(p),0) for s,p in zip(seeds,best_exp_performance) if s in non_converged_seeds}

    if best_sim_stats=='converged' and len(converged_seeds)==0:
        best_sim_stats = None

    if best_sim_stats is not None:
        best_stats_non_flat_neur_outputs = {}
        best_stats_non_flat_neur_states = {}
        best_stats_non_flat_motors = {}
        if sim.num_agents == 2:
            best_stats_genetic_distance = {}

        best_stats_seeds = converged_seeds if best_sim_stats=='converged' else seeds

        for s in best_stats_seeds:
            s_exp_dir = seed_exp_dir[s]
            performance, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(s_exp_dir, quiet=True)
            data_record = data_record_list[sim_idx]   
            best_stats_non_flat_neur_outputs[s] = get_non_flat_neuron_data(data_record, 'agents_brain_output')
            best_stats_non_flat_neur_states[s] = get_non_flat_neuron_data(data_record, 'agents_brain_state')
            best_stats_non_flat_motors[s] = get_non_flat_neuron_data(data_record, 'agents_motors')
            if sim.num_agents == 2:
                best_stats_genetic_distance[s] = data_record['genotype_distance']

    if print_stats:
        # print('Selected evo: {}'.format(last_evo_file))
        # print('Num seeds:', len(best_exp_performance))
        # print('Stats:', stats.describe(best_exp_performance))
        print(f'Converged ({len(converged_seeds)}):', converged_seeds)
        print('\tConverged seed/error:', conv_seeds_err)
        print('\tNon Converged seed/error:', non_conv_seeds_err)
        # print(f'Non converged ({len(non_converged_seeds)}):', non_converged_seeds)

        if best_sim_stats:
            if sim.num_agents == 2:
                # print('Genetic distances:')
                # for s in best_stats_seeds:
                #     print(f'\tSeed {str(s).zfill(3)}: {best_stats_genetic_distance[s]}')
                print(f'Average genetic distance: {np.mean(list(best_stats_genetic_distance.values()))}')            
            print(f'Non flat neurons outputs for each agent (min-max): {min_max_flat_elements(best_stats_non_flat_neur_outputs.values())}')
            # for s in best_stats_seeds:
            #     print(f'\tSeed {str(s).zfill(3)}: {best_stats_non_flat_neur_outputs[s]}')            
            print(f'Non flat neurons states for each agent (min-max): {min_max_flat_elements(best_stats_non_flat_neur_states.values())}')
            # for s in best_stats_seeds:
            #     print(f'\tSeed {str(s).zfill(3)}: {best_stats_non_flat_neur_states[s]}')

    if export_to_csv:
        # save file to csv
        f_name = os.path.join(base_dir,'gen_seeds_error.csv')
        print('saving csv:', f_name)
        all_gen_best_performances = np.transpose(np.array(all_gen_best_performances))
        num_agents, num_gen, num_seeds = all_gen_best_performances.shape
        if num_agents==1:
            all_gen_best_performances = all_gen_best_performances[0,:,:]
            seeds_str = [f'seed_{str(s)}' for s in seeds]            
        else:
            assert num_agents==2
            # num_agents, num_gen, num_seeds -> num_gen, num_seeds, num_agents
            all_gen_best_performances = np.moveaxis(all_gen_best_performances, 0, 2) 
            all_gen_best_performances = np.reshape(all_gen_best_performances,  (num_gen, 2*num_seeds))
            seeds_str = [ [f'seed_{str(s)}A', f'seed_{str(s)}B'] for s in seeds]
            seeds_str = [i for g in seeds_str for i in g] # flattening
        df = pd.DataFrame(all_gen_best_performances, columns=seeds_str)  
        df.to_csv(f_name, index=False)

    if plot:
        plot_best_exp_performance(best_exp_performance, seeds)
        if best_sim_stats:
            if sim.num_agents == 2:
                bar_plot_seeds_data_value(best_stats_genetic_distance, 'Genetic distance')
            bar_plot_seeds_data_list(best_stats_non_flat_neur_outputs, 'Non flat neurons outputs')
            # bar_plot_seeds_data_list(best_stats_non_flat_neur_states, 'Non flat neurons states')
            bar_plot_seeds_data_list(best_stats_non_flat_motors, 'Non flat motors')

    return converged_seeds


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze results'
    )

    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--print_values', action='store_true', default=False, help='Whether to export results to csv in same dir')
    parser.add_argument('--best_sim_stats', type=str, default=None, choices=[None, 'converged', 'all'], help='Whether to run best simulation stats (non-flat neurons/motors, similarities) and on which seeds')
    parser.add_argument('--plot', action='store_true', default=False, help='Whether to export results to csv in same dir')
    parser.add_argument('--csv', action='store_true', default=False, help='Whether to export results to csv in same dir')

    args = parser.parse_args()

    get_last_performance_seeds(
        base_dir=args.dir, 
        print_stats=True, 
        print_values=args.print_values, 
        plot=args.plot, 
        export_to_csv=args.csv,
        best_sim_stats=args.best_sim_stats,
    )
