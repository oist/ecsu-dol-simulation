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

def get_last_performance_runs(base_dir, print_values, print_stats, plot, export_to_csv):
    exp_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('seed_')])
    best_exp_performance = []  # the list of best performances of last generation for all seeds
    all_gen_best_performances = [] # all best performances for all seeds for all generations
    last_evo_file = None
    seeds = []
    for exp in exp_dirs:
        exp_dir = os.path.join(base_dir, exp)
        if last_evo_file is None:
            evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
            if len(evo_files) == 0:
                # no evo files
                continue
            last_evo_file = evo_files[-1]
            print('Selected evo: {}'.format(last_evo_file))
        evo_file = os.path.join(exp_dir, last_evo_file)
        if not os.path.isfile(evo_file):
            continue
        with open(evo_file) as f_in:
            sim_json_filepath = os.path.join(exp_dir, 'simulation.json')    
            sim = Simulation.load_from_file(sim_json_filepath)
            exp_evo_data = json.load(f_in)
            seeds.append(exp_evo_data['random_seed'])
            gen_best_perf = np.array(exp_evo_data['best_performances'])
            gen_best_perf = sim.max_mean_distance - gen_best_perf

            # make sure it's monotonic increasing(otherwise there is a bug)
            # assert all(gen_best_perf[i] <= gen_best_perf[i+1] for i in range(len(gen_best_perf)-1))

            perf_index = lambda a: '|'.join(['{:.5f}'.format(x) for x in a])

            last_best_performance = gen_best_perf[-1]            
            if print_values:
                print('{} {}'.format(exp, perf_index(last_best_performance)))
            best_exp_performance.append(last_best_performance)
            all_gen_best_performances.append(gen_best_perf)

    if print_stats:        
        print(stats.describe(best_exp_performance))

    if export_to_csv:
        # save file to csv
        f_name = os.path.join(base_dir,'gen_seeds_perf.csv')
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
        # print("seeds:",seeds)
        fig, ax = plt.subplots()
        ind = np.arange(len(seeds))        
        num_bars = len(best_exp_performance[0])
        width = 0.7 / num_bars
        for p in range(num_bars):
            p_series = [b[p] for b in best_exp_performance]
            x_pos = ind + p * width
            if num_bars == 1:
                x_pos = x_pos + width / 2  # center bar on ticks if there is only one bar
            ax.bar(x_pos, p_series, width)
        ax.set_xticks(ind + 0.7 / 2)
        ax.set_xticklabels(seeds)
        # plt.ylim(4500, 5000)
        plt.xlabel('Seeds')
        plt.ylabel('Performance')
        plt.show()
    # return dict(zip(seeds, best_exp_performance))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze results'
    )

    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--print_values', type=bool, default=False, help='Whether to export results to csv in same dir')
    parser.add_argument('--print_stats', type=bool, default=False, help='Whether to export results to csv in same dir')
    parser.add_argument('--plot', type=bool, default=True, help='Whether to export results to csv in same dir')
    parser.add_argument('--csv', type=bool, default=False, help='Whether to export results to csv in same dir')

    args = parser.parse_args()

    get_last_performance_runs(
        args.dir, args.print_values, 
        args.print_stats, args.plot, args.csv
    )
