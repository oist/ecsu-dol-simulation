"""
TODO: Missing module docstring
"""

import matplotlib.pyplot as plt
from dol.simulation import MAX_MEAN_DISTANCE
import numpy as np


def plot_performances(evo, log=False, only_best=False):
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Agent Performances")
    ax = fig.add_subplot(1,1,1)
    if log:
        ax.set_yscale('log')
    best_perf = MAX_MEAN_DISTANCE - np.array(evo.best_performances)
    if only_best:
        ax.plot(best_perf, label='Best')
    else:
        ax.plot(best_perf, label='Best')
        avg_perf = MAX_MEAN_DISTANCE - np.array(evo.avg_performances)
        worse_perf = MAX_MEAN_DISTANCE - np.array(evo.worst_performances)
        ax.plot(avg_perf, label='Avg')
        ax.plot(worse_perf, label='Worst')    
    plt.legend()
    plt.show()

def plot_data_scatter(data_record, key, log=False):
    exp_data = data_record[key]
    num_trials = len(exp_data)
    fig = plt.figure(figsize=(10, 6))
    title = key.replace('_',' ').title() + " (Scattter)"
    fig.suptitle(title)
    for t in range(num_trials):  
        trial_data = exp_data[t]  
        num_agents = len(trial_data)   
        for a in range(num_agents):
            ax = fig.add_subplot(num_agents, num_trials, (a*num_trials)+t+1) # projection='3d'
            if log:
                ax.set_yscale('log')            
            agent_trial_data = trial_data[a]
            # initial position
            ax.scatter(agent_trial_data[0][0], agent_trial_data[0][1], color='orange', zorder=1) 
            ax.plot(agent_trial_data[:, 0], agent_trial_data[:, 1], zorder=0)
    plt.show()    

def plot_data_time(data_record, key, trial='all', log=False):
    exp_data = data_record[key]
    num_trials = len(exp_data) if trial == 'all' else 1
    fig = plt.figure(figsize=(10, 6))
    title = key.replace('_',' ').title() + " (Time)"
    fig.suptitle(title)
    for t in range(num_trials):
        trial_data = exp_data[t] if trial == 'all' else exp_data[trial-1]
        if type(trial_data) is list:
            num_agents = len(trial_data)   
            for a in range(num_agents):
                ax = fig.add_subplot(num_agents, num_trials, (a*num_trials)+t+1)
                if log: ax.set_yscale('log')            
                agent_trial_data = trial_data[a]
                for n in range(agent_trial_data.shape[1]):
                    ax.plot(agent_trial_data[:, n], label='data {}'.format(n+1))                    
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper right')
        else:
            ax = fig.add_subplot(1, num_trials, t+1)
            if log: ax.set_yscale('log')            
            ax.plot(trial_data)                    
    
    plt.show()

def plot_motor_time(sim, data_record, key, trial='all', log=False):
    exp_data = data_record[key]
    num_trials = len(exp_data) if trial == 'all' else 1
    fig = plt.figure(figsize=(10, 6))
    title = key.replace('_',' ').title() + " (Time)"
    fig.suptitle(title)
    for t in range(num_trials):
        trial_index = t if trial == 'all' else trial-1
        trial_data = exp_data[trial_index]
        if type(trial_data) is list:
            num_agents = len(trial_data)   
            for a in range(num_agents):
                ax = fig.add_subplot(num_agents, num_trials, (a*num_trials)+t+1)
                if log: ax.set_yscale('log')            
                agent_trial_data = trial_data[a]
                for n in range(agent_trial_data.shape[1]):
                    color= ['blue', 'orange'][n]
                    # if sim.exclusive_motors_threshold is not None and data_record['agents_motors_control_indexes'][trial_index][n]!=a: 
                    #     # discard motor if not wired
                    #     continue
                    ax.plot(agent_trial_data[:, n], label='data {}'.format(n+1),color=color)                    
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper right')
        else:
            ax = fig.add_subplot(1, num_trials, t+1)
            if log: ax.set_yscale('log')            
            ax.plot(trial_data)                    
    
    plt.show()

def plot_data_time_multi_keys(data_record, keys, title, log=False):    
    num_trials = len(data_record[keys[0]])
    fig = plt.figure(figsize=(10, 6))
    if title is not None:
        fig.suptitle(title)
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_trials, t+1)
        if log: ax.set_yscale('log')            
        for k in keys:    
            trial_data = data_record[k][t]                        
            ax.plot(trial_data)                    
    plt.show()

def plot_genotype_similarity(evo, sim):
    from sklearn.metrics.pairwise import pairwise_distances
    population = evo.population
    if len(evo.population)>1:
        population = np.concatenate(evo.population)
        population = np.expand_dims(population, 0)
    similarity = 1 - pairwise_distances(population[0])
    # print(similarity.shape)
    plt.imshow(similarity)
    plt.colorbar()
    plt.show()

    if sim.num_random_pairings==0:
        # genotypes evolved in pairs
        similarity = np.zeros((1, len(population)))
        for i,pair in enumerate(population):
            a,b = np.array_split(pair, 2)  
            similarity[0][i] = 1 - np.linalg.norm(a-b)
        plt.imshow(similarity)
        plt.colorbar()
        plt.show()            

def plot_results(evo, sim, trial, data_record):

    if trial is None:
        trial = 'all'
    
    if evo is not None:
        plot_performances(evo, log=True)   
        plot_performances(evo, log=False, only_best=True)   
        plot_genotype_similarity(evo, sim)
    
    # scatter agents
    plot_data_scatter(data_record, 'agents_brain_output')
    plot_data_scatter(data_record, 'agents_brain_state')    
    
    # time agents
    plot_data_time(data_record, 'agents_brain_input', trial)
    plot_data_time(data_record, 'agents_brain_output', trial)
    # plot_data_time(data_record, 'agents_brain_state', trial)
    # plot_data_time(data_record, 'agents_derivatives', trial)
    plot_motor_time(sim, data_record, 'agents_motors', trial)

    # time tracker
    # plot_data_time(data_record, 'tracker_wheels', trial)
    # plot_data_time(data_record, 'tracker_velocity', trial)
    plot_data_time(data_record, 'tracker_signals', trial)

    # time target
    # plot_data_time(data_record, 'target_velocity', trial)    

    # time tracker & target
    plot_data_time_multi_keys(
        data_record, 
        keys=['tracker_position', 'target_position'], 
        title="Tracker and Target"
    )

    # delta tracker target (distances)
    plot_data_time(data_record, 'delta_tracker_target', trial)    

    # plot_genotype_similarity(evo, sim)


def test_plot():
    from dol import simulation    
    sim, data_record = simulation.get_simulation_data_from_random_agent()
    plot_results(None, sim, data_record)


if __name__ == "__main__":
    test_plot()


