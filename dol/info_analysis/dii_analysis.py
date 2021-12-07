import os

import numpy as np
from tqdm import tqdm
import pickle
from dol.info_analysis.dii import DII
from dol.info_analysis import infodynamics
from dol.info_analysis.info_analysis import build_info_analysis_from_experiments
from dol.info_analysis.plots import box_plot
from dol.info_analysis.info_utils import interpret_observed_effect_size, show_descriptive_stats
from joblib import Parallel, delayed
from collections import defaultdict
from scipy.stats import spearmanr, ranksums



def perform_analysis(IA, agent_nodes, conditioning_node, powerset=False):

    data = IA.data
    ouput_dir = IA.output_dir
    num_cores = IA.num_cores

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
    conditioning_node
    'MI powerset'
    'Cond MI powerset'
    'MI overall'
    'CMI overall'
    'Synergy powerset'
    'Synergy overall'
    '''

    # # SpearmanCorr for ['MI overall', 'CMI overall', 'Synergy overall']        
    for x in ['MI overall', 'CMI overall', 'Synergy overall']:            
        for sim_type in sim_types:
            target = [ # average target-tracker distance for all seeds (averaged across timesteps and then trials)        
                sim_type_results[sim_type][s][conditioning_node] 
                for s in range(num_seeds)
            ]            
            measure_values = [
                sim_type_results[sim_type][s][x] 
                for s in range(num_seeds)
            ]
            [r, p] = spearmanr(measure_values, target) # where x and delta are arrays of num_seeds values            
            print('-------')
            print(f'SpearmanCorr {sim_type} - {x} - {conditioning_node}')
            print('r:', r)
            print('p:', p)
    
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
            interpret_observed_effect_size(effectSize, 2), ')')
        show_descriptive_stats(sims_metric_data[0], sim_type[0])
        show_descriptive_stats(sims_metric_data[1], sim_type[1])        


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
    target_trials = np.zeros(num_trials)
    if powerset:
        synergy_powerset = np.zeros(num_trials)
        
    for t in range(num_trials):
        agent1 = np.concatenate([sim_data[node][t,0,:,:] for node in agent_nodes], axis=1)
        agent2 = np.concatenate([sim_data[node][t,1,:,:] for node in agent_nodes], axis=1)
        target = sim_data[conditioning_node][t]
        target_trials[t] = target.mean() # average across time steps
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
        conditioning_node: target_trials.mean(),
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


if __name__ == "__main__":

    # agent_nodes = ['agents_brain_input', 'agents_brain_state', 'agents_brain_output']
    agent_nodes = ['agents_sensors', 'agents_brain_output'] # alife settings
    conditioning_node = 'delta_tracker_target'

    powerset = False

    IA = build_info_analysis_from_experiments()
    perform_analysis(IA, agent_nodes, conditioning_node, powerset=False)
    
