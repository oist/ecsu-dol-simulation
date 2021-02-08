import os
import json
import matplotlib.pyplot as plt
from scipy import stats
import sys
import numpy as np
from dol.simulation import MAX_MEAN_DISTANCE

def get_last_performance_runs(base_dir, plot=True):    
    # base_dir = 'data/transfer_entropy/MAX'
    exp_dirs = sorted(os.listdir(base_dir))
    best_exp_performance = []
    last_evo_file = None
    seeds = []
    for exp in exp_dirs:
        exp_dir = os.path.join(base_dir, exp)
        if last_evo_file is None:
            evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
            if len(evo_files)==0:
                # no evo files
                continue
            last_evo_file = evo_files[-1]
            print('Selected evo: {}'.format(last_evo_file))
        evo_file = os.path.join(exp_dir, last_evo_file)
        if not os.path.isfile(evo_file):
            continue
        with open(evo_file) as f_in:
            exp_evo_data = json.load(f_in)
            seeds.append(exp_evo_data['random_seed'])
            gen_best_perf = np.array(exp_evo_data['best_performances'])
            gen_best_perf = MAX_MEAN_DISTANCE - gen_best_perf
            
            # make sure it's monotonic increasing(otherwise there is a bug)
            # assert all(gen_best_perf[i] <= gen_best_perf[i+1] for i in range(len(gen_best_perf)-1))

            print_stats = lambda a : '|'.join(['{:.5f}'.format(x) for x in a])
            
            last_best_performance = gen_best_perf[-1]
            print('{} {}'.format(exp, print_stats(last_best_performance)))
            best_exp_performance.append(last_best_performance)
    print(stats.describe(best_exp_performance))
    if plot:
        print("seeds:",seeds)
        fig, ax = plt.subplots()
        ind = np.arange(len(seeds))
        width = 0.35
        for p in range(len(best_exp_performance[0])):
            p_series = [b[p] for b in best_exp_performance]
            x_pos = ind + p * width
            ax.bar(x_pos, p_series, width)
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(seeds)
        # plt.ylim(4500, 5000)
        plt.xlabel('Seeds')
        plt.ylabel('Performance')
        plt.show()
    return dict(zip(seeds, best_exp_performance))

if __name__ == "__main__":
    assert len(sys.argv)==2, "You need to specify the directory with the various runs to analyze"    
    base_dir = sys.argv[1]
    get_last_performance_runs(base_dir)