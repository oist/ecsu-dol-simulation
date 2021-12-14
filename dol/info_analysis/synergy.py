#! /usr/bin/env python 3

import sys
import os
import numpy as np
from tqdm import tqdm
from typing import Counter
from scipy.spatial.distance import pdist, squareform
from dol.info_analysis.info_analysis import InfoAnalysis, build_info_analysis_from_experiments
from dol.info_analysis.infodynamics import compute_mi, compute_cond_mi
from dol.info_analysis import info_utils

def compute_synergy(IA, agent_nodes):

    results = {} 
    # overall results sim_type -> sim_type_result
    # where sym_type_result is a dictionary: measure -> data (num_seed)
    nodes_measures = ['condMultVarMI', 'multVarMI', 'coinformation']
    dist_measures = ['trackerTargetDistMean', 'trackerTargetDistStd']
    results_measures = nodes_measures + dist_measures

    sim_type_num_seeds = {
        sim_type: len(seed_sim_data) # how many converged seeds in each sim type
        for sim_type, seed_sim_data in IA.data.items()
    }

    all_sim_same_converged_seeds = len(set(sim_type_num_seeds.values())) == 1 # True
    min_num_converged_seeds = min(sim_type_num_seeds.values())

    assert IA.bootstrapping or all_sim_same_converged_seeds, \
        f"Cannot compute statistics without bootstrapping if sim_type have different number of converged seeds: {sim_type_num_seeds}"
    
    assert not IA.bootstrapping or IA.num_seeds_boostrapping < min_num_converged_seeds, \
        f"You specified a num_seed_stats value that is >= to min number of converged seeds: {min_num_converged_seeds}"

    for sim_type, seed_sim_data in IA.data.items():

        num_seeds = len(seed_sim_data)  # number of converged seeds

        sim_type_results = {
            measure: np.zeros((num_seeds, IA.num_trials)) # num_seeds x num_trials (e.g., 100x4) eventually will turn into a num_seeds array (mean/std across columns)
            for measure in results_measures
        }

        for s, sim_data in enumerate(seed_sim_data.values()):

            # s is the seed index
            # sim_data is the dictionary with the results from the simulation

            delta_tracker_target = sim_data['delta_tracker_target'] # (num_trials, num_data_points)

            for t in range(IA.num_trials):
                # print('Trial # ', (t + 1))
                agent1 = np.concatenate([sim_data[node][t,0,:,:] for node in agent_nodes], axis=1)
                agent2 = np.concatenate([sim_data[node][t,1,:,:] for node in agent_nodes], axis=1)
                target_pos = sim_data['target_position'][t]
                sim_type_results['condMultVarMI'][s,t] = condMultVarMI = \
                    compute_cond_mi(agent1, agent2, np.expand_dims(target_pos, axis = 1))
                sim_type_results['multVarMI'][s,t] = multVarMI = compute_mi(agent1, agent2)
                sim_type_results['coinformation'][s,t] = condMultVarMI - multVarMI  #### a.k.a interaction information, net synergy, and integration														
                sim_type_results['trackerTargetDistMean'][s,t] = delta_tracker_target[t].mean()
                sim_type_results['trackerTargetDistStd'][s,t] = delta_tracker_target[t].std()
                
        # compute mean across trials
        # all variables will be 1-dim array with num_seeds elements
        for measure in sim_type_results:
            if measure == 'trackerTargetDistStd':
                # we take the std across trials for std
                sim_type_results[measure] = sim_type_results[measure].std(axis=1)	
            else:
                # we take the mean across trials for all other values
                sim_type_results[measure] = sim_type_results[measure].mean(axis=1) # 

        results[sim_type] = sim_type_results

    info_measures = {
        'condMultVarMI': 'Multivariate Conditional Mutual Information',				
        'multVarMI': 'Multivariate Mutual Information',
        'coinformation': 'Net-Synergy'
    }

    if IA.plot:
        for measure, label in info_measures.items():
            results_measure_sim_types = [results[sim_type][measure] for sim_type in IA.simulation_types]
            IA.plotBoxPlotList(results_measure_sim_types, IA.simulation_types, label, label)                        


    if IA.bootstrapping:			
        # condMultVarMI = [results[sim_type]['condMultVarMI'] for sim_type in IA.simulation_types] # num_sim_type rows x converged_seeds_in_sim_type
        # multVarMI = [results[sim_type]['multVarMI'] for sim_type in IA.simulation_types]
        # coinformation = [results[sim_type]['coinformation'] for sim_type in IA.simulation_types]			
        # TODO: boostrapping (random sampling with replacement)

        stats_factory = lambda: {
            'h': np.zeros(IA.bootstrapping_runs),
            'p': np.zeros(IA.bootstrapping_runs),
            'eta': np.zeros(IA.bootstrapping_runs),
            'epsilon': np.zeros(IA.bootstrapping_runs),
            'post_hoc_stats': np.zeros((IA.bootstrapping_runs,IA.num_sim_types, 3))
        }

        boostrapping_stats = {
            measure: stats_factory()
            for measure in info_measures
        }
        
        sim_type_seed_idx_counter = {
            sim_type: Counter()
            for sim_type in IA.simulation_types
        }

        for b in tqdm(range(IA.bootstrapping_runs)):	

            for measure, label in info_measures.items():					

                selected_stat_indexes = np.zeros((IA.num_sim_types, IA.num_seeds_boostrapping), dtype="int")
                selected_stat = np.zeros((IA.num_sim_types, IA.num_seeds_boostrapping))
                
                for i, sim_type in enumerate(IA.simulation_types):
                    indexes = list(range(len(results[sim_type][measure])))
                    selected_stat_indexes[i] = IA.rs.choice(indexes, size=IA.num_seeds_boostrapping, replace=True)
                    selected_stat[i] = np.take(results[sim_type][measure], selected_stat_indexes[i])

                for sim_index, sim_type_seed_idx_choices in enumerate(selected_stat_indexes):
                    sim_type = IA.simulation_types[sim_index]
                    sim_type_seed_idx_counter[sim_type].update(sim_type_seed_idx_choices)

                selected_stat = info_utils.normalize_data(selected_stat, IA.norm_type) 
            
                ################# We might want to check whether data follows normal distribution and if positive apply parametric tests instead.

                # IA.checkDataNormality(selected_stat.flatten().tolist(), label)

                h, p, eta, epsilon, post_hoc_stats = IA.performKruskalWallis_n_PosthocWilcoxonTest(selected_stat, label)

                boostrapping_stats_measure = boostrapping_stats[measure]
                boostrapping_stats_measure['h'][b] = h
                boostrapping_stats_measure['p'][b] = p
                boostrapping_stats_measure['eta'][b] = eta
                boostrapping_stats_measure['epsilon'][b] = epsilon
                boostrapping_stats_measure['post_hoc_stats'][b] = post_hoc_stats

        if IA.plot:
            IA.plot_seed_choices(sim_type_seed_idx_counter)

        for measure, label in info_measures.items():
            # TODO: maybe only print those stats for which mean(p) < bonferroni ...
            boostrapping_stats_measure = boostrapping_stats[measure]
            print(label)
            for sub_measure, data in boostrapping_stats_measure.items():					
                info_utils.show_descriptive_stats(data, sub_measure)

            
    else:
        # no bootstrpping
        # following arrays have shape num_seeds x num_sim_types (e.g., 100 x 3)
        condMultVarMI = np.array([results[sim_type]['condMultVarMI'] for sim_type in IA.simulation_types]).T
        multVarMI = np.array([results[sim_type]['multVarMI'] for sim_type in IA.simulation_types]).T
        coinformation = np.array([results[sim_type]['coinformation'] for sim_type in IA.simulation_types]).T
        
        condMultVarMI = info_utils.normalize_data(condMultVarMI, IA.norm_type) 
        multVarMI = info_utils.normalize_data(multVarMI, IA.norm_type)
        coinformation = info_utils.normalize_data(coinformation, IA.norm_type)

        ################# We might want to check whether data follows normal distribution and if positive apply parametric tests instead.

        # IA.checkDataNormality(condMultVarMI.flatten().tolist(), 'Multivariate Conditional Mutual Information')
        # IA.checkDataNormality(multVarMI.flatten().tolist(), 'Multivariate Mutual Information')
        # IA.checkDataNormality(coinformation.flatten().tolist(), 'Net-Synergy')			

        IA.performKruskalWallis_n_PosthocWilcoxonTest(condMultVarMI, f'Multivariate Conditional Mutual Information')
        IA.performKruskalWallis_n_PosthocWilcoxonTest(multVarMI, f'Multivariate Mutual Information')
        IA.performKruskalWallis_n_PosthocWilcoxonTest(coinformation, f'Net-Synergy')		

        print('\n\n Spearman Correlation Based on Target-Tracker Mean Distance')

        for sim_type, sim_type_results in results.items():
            cond_mult_coinfo_mean = np.array([sim_type_results[m] for m in nodes_measures]).T
            IA.computeSpearmanCorr(
                cond_mult_coinfo_mean, 
                sim_type_results['trackerTargetDistMean'], 
                sim_type + '_Mean', 
                'Mean Target-Tracker Disatnce'
            )  ##### 1 : z-scored   2 : [0 .. 1] scaled

        print('\n\n Spearman Correlation Based on Target-Tracker SD Distance')

        for sim_type, sim_type_results in results.items():
            cond_mult_coinfo_mean = np.array([sim_type_results[m] for m in nodes_measures]).T
            IA.computeSpearmanCorr(
                cond_mult_coinfo_mean, 
                sim_type_results['trackerTargetDistStd'], 
                sim_type + '_SD', 
                'SD Target-Tracker Disatnce'
            )		

def computeDistanceMetricsForSpecificSeed(IA, agent_nodes, whichSetting, whichSeed, trial_idx, whichDistance):
    if not whichSeed in list(set(os.listdir(IA.sim_type_path[whichSetting]))):
        print(whichSeed, '  Is Not a Valid Seed')				
        sys.exit()

    data = IA.data[whichSetting][whichSeed]

    agent1 = np.concatenate([data[node][trial_idx,0,:,:] for node in agent_nodes], axis=1)
    agent2 = np.concatenate([data[node][trial_idx,1,:,:] for node in agent_nodes], axis=1)
    agentsM = np.concatenate((agent1, agent2), axis = 1).T

    agentsM = info_utils.normalize_data(agentsM, IA.norm_type)			

    agentsM = squareform(pdist(agentsM, whichDistance))

    labels = []
    cnt = 0
    for i in range(agentsM.shape[0]):
        if i < 6:
            labels.append('Node1_' + str(cnt + 1))
        else:
            if i == 6:
                cnt = 0
            labels.append('Node2_' + str(cnt + 1))
        cnt += 1

    IA.generateHeatMap(
        agentsM, 
        labels, 
        f'{whichSetting} {whichSeed} Trial {trial_idx+1} {whichDistance} Distance'
    )
        

if __name__ == "__main__":
    # read appropriate arguments from command line
    IA = build_info_analysis_from_experiments()

    agent_nodes = ['agents_brain_input', 'agents_brain_state', 'agents_brain_output']
    
    # main computation script
    compute_synergy(IA, agent_nodes)

    ''' 
    correlation = 1 - corr(x, y)  AND  canberra = \sum_i (abs(x_i - y_i))/(abs(x_i) + abs(y_i))
    '''
    # distanceMetrics = ['cosine', 'correlation', 'euclidean', 'cityblock', 'canberra']   
    # distanceMetrics = ['correlation']   
    # for metric in distanceMetrics:
    # 	IA.computeDistanceMetricsForSpecificSeed(IA, agent_nodes, 'individual', 'seed_001', 0, metric)
