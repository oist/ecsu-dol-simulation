import os
from measures.plots import box_plot
import numpy as np
from tqdm import tqdm
import pickle
from measures.dii import DII
from measures import infodynamics
from joblib import Parallel, delayed
from collections import defaultdict
from scipy.stats import spearmanr, ranksums

from dol.info_analysis.info_utils import interpretObservedEffectSize, showDescriptiveStatistics

def load_data_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)		
    return data

def compute_info_values(sim_data, agent_nodes, conditioning_node, powerset=False):
    """compute info measures for data of a specific simulation type and a specific seed

    Args:
        sim_data ([type]): [description]
        agent_nodes ([type]): [description]
        conditioning_node ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    # we know num trials is first dim in data
    # TODO: fix this to make it more robust
    num_trials = sim_data[agent_nodes[0]].shape[0]

    mi_trials = np.zeros(num_trials)
    mi_overall_trials = np.zeros(num_trials)
    cond_mi_trials = np.zeros(num_trials)
    cond_mi_overall_trials = np.zeros(num_trials)
    synergy_overall = np.zeros(num_trials)
    if powerset:
        synergy_powerset = np.zeros(num_trials)
        
    for t in range(num_trials):
        agent1 = np.concatenate([sim_data[node][t,0,:,:] for node in agent_nodes], axis=1)
        agent2 = np.concatenate([sim_data[node][t,1,:,:] for node in agent_nodes], axis=1)
        target = sim_data[conditioning_node][t]
        dii = DII(agent1, agent2, target)        
        overall_mi, overall_cond_mi = dii.compute_dii_overall()        
        if powerset:
            mi_matrix, cond_mi_matrix = dii.compute_dii_powerset() 
            mi_trials[t] = mi_matrix.mean()                
            cond_mi_trials[t] = cond_mi_matrix.mean()
            synergy_powerset[t] = cond_mi_trials[t] - mi_trials[t]
        
        cond_mi_overall_trials[t] = overall_cond_mi        
        mi_overall_trials[t] = overall_mi
        synergy_overrall_trial = cond_mi_overall_trials[t] - mi_overall_trials[t]
        synergy_overall[t] = synergy_overrall_trial

    result = {
        'MI overall': mi_overall_trials.mean(),
        'CMI overall': cond_mi_overall_trials.mean(),
        'Synergy overall': synergy_overall.mean()
    }
    if powerset:
        result.update(
            {
                'MI powerset': mi_trials.mean(),
                'Cond MI powerset': cond_mi_trials.mean(),
                'Synergy powerset': synergy_powerset.mean()
            }
        )

    return result



def perform_analysis(data, ouput_dir, num_cores, agent_nodes, conditioning_node, powerset=False):

    sim_type_results = defaultdict(lambda: defaultdict(dict))
    # sim type -> seed_num -> result 
    # keys are the simulation types
    # values is a dict mapping seed to result

    sim_types = list(data.keys())
    first_sim_type = sim_types[0]
    num_seeds = len(data[first_sim_type])

    for sim_type, seed_sim_data in data.items():

        if num_cores == 1:
            for s, sim_data in enumerate(tqdm(seed_sim_data.values())):
                sim_type_results[sim_type][s] = compute_info_values(sim_data, agent_nodes, conditioning_node, powerset)
        else:
            seeds_results = Parallel(n_jobs=num_cores)(
                delayed(compute_info_values)(sim_data, agent_nodes, conditioning_node, powerset)
                for sim_data in tqdm(seed_sim_data.values())
            )
            for s, results in enumerate(seeds_results):
                sim_type_results[sim_type][s] = results

    output_file = None
    
    result_metrics = list(sim_type_results[first_sim_type][0].keys())
    '''
    'MI powerset'
    'Cond MI powerset'
    'MI overall'
    'CMI overall'
    'Synergy powerset'
    'Synergy overall'
    '''

    
    # # SpearmanCorr for ['MI overall', 'CMI overall', 'Synergy overall']        
    '''
    for x in ['MI overall', 'CMI overall', 'Synergy overall']:            
        delta = None # average target-tracker distance for all seeds (averaged across timesteps and then trials)
        # self.delta_target = target_position - self.position
        # delta_abs = np.abs(self.delta_target)
        [r, p] = spearmanr(x, delta) # where x and delta are arrays of num_seeds values
    '''
    
    for metric in result_metrics:

        if ouput_dir is not None:
            output_file = os.path.join(ouput_dir, f'box_plot_{metric}.pdf')

        sims_metric_data = [
            [
                sim_type_results[sim_type][s][metric] 
                for s in range(num_seeds)
            ] 
            for sim_type in sim_types
        ]

        box_plot(
            data = sims_metric_data,
            labels = sim_types, 
            title = '', # Dyadic Integrated Information
            ylabel = metric,
            output_file = output_file
        )

        print('-------------')
        print(metric)
        print(f'sim_types: {sim_types}')
        sW, pW = ranksums(sims_metric_data[0], sims_metric_data[1])        
        effectSize = abs(sW/np.sqrt(len(sims_metric_data[0])))
        print(sim_type[0], ' vs. ', sim_type[1], '  s = ', sW, '  p = ', pW, '  effect-size = ', effectSize, '(', \
            interpretObservedEffectSize(effectSize, 2), ')')
        showDescriptiveStatistics(sims_metric_data[0], sim_type[0])
        showDescriptiveStatistics(sims_metric_data[1], sim_type[1])

    infodynamics.shutdownJVM
        

if __name__ == "__main__":
    import argparse

    # agent_nodes = ['agents_brain_input', 'agents_brain_state', 'agents_brain_output']
    agent_nodes = ['agents_sensors', 'agents_brain_output'] # alife settings
    conditioning_node = 'delta_tracker_target'

    parser = argparse.ArgumentParser(
        description='DII Simulation Analysis'
    )

    parser.add_argument('--pickle_path', type=str, required=True, help='Pickle path')
    parser.add_argument('--num_cores', type=int, default=1, help='Number of cores to used (defaults to 1)')
    parser.add_argument('--powerset', action='store_true', default=False, help='Whether to compute powerset metrics')
    parser.add_argument('--output_dir', type=str, default=None, help='Output dir where to save plots (defaults to None: disply plots to screen)')

    args = parser.parse_args()

    pickle_path = args.pickle_path 
    # pickle_path = '/Users/fedja/Code/ECSU/Evolution/dol-simulation/results/alife21_5seeds.pickle'
    # args.num_cores = 5
    # args.output_dir = './results/alife21_5'
    data = load_data_from_pickle(pickle_path)

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    perform_analysis(data, args.output_dir, args.num_cores, agent_nodes, conditioning_node, args.powerset)
