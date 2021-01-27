"""
TODO: Missing module docstring
"""

import os
import matplotlib.pyplot as plt
from dol.simulation import Simulation
from dol import gen_structure
from dol import utils
import numpy as np
from numpy.random import RandomState
from pyevolver.evolution import Evolution


def plot_performances(evo, log=False):
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Agent Performances")
    ax = fig.add_subplot(1,1,1)
    if log:
        ax.set_yscale('log')
    ax.plot(evo.best_performances, label='Best')
    ax.plot(evo.avg_performances, label='Avg')
    ax.plot(evo.worst_performances, label='Worst')    
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

def plot_data_time(data_record, key, log=False):
    exp_data = data_record[key]
    num_trials = len(exp_data)
    fig = plt.figure(figsize=(10, 6))
    title = key.replace('_',' ').title() + " (Time)"
    fig.suptitle(title)
    for t in range(num_trials):
        trial_data = exp_data[t]          
        if type(trial_data) is list:
            num_agents = len(trial_data)   
            for a in range(num_agents):
                ax = fig.add_subplot(num_agents, num_trials, (a*num_trials)+t+1)
                if log: ax.set_yscale('log')            
                agent_trial_data = trial_data[a]
                for n in range(agent_trial_data.shape[1]):
                    ax.plot(agent_trial_data[:, n], label='Output of n{}'.format(n+1))                    
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

'''
def plot_genotype_similarity(evo, sim):
    from sklearn.metrics.pairwise import pairwise_distances
    population = evo.population
    similarity = 1 - pairwise_distances(population)
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
'''

def plot_results(evo, sim, data_record):
    
    if evo is not None:
        plot_performances(evo)    
    
    # scatter agents
    # plot_data_scatter(data_record, key='agents_brain_output')
    # plot_data_scatter(data_record, key='agents_brain_state')    
    
    # time agents
    plot_data_time(data_record, key='agents_brain_input')
    # plot_data_time(data_record, key='agents_brain_output')
    # plot_data_time(data_record, key='agents_brain_state')
    # plot_data_time(data_record, key='agents_derivatives')

    # time tracker
    # plot_data_time(data_record, key='tracker_wheels')
    # plot_data_time(data_record, key='tracker_velocity')
    plot_data_time(data_record, key='tracker_signals')

    # time target
    # plot_data_time(data_record, key='target_velocity')    

    # time tracker & target
    plot_data_time_multi_keys(
        data_record, 
        keys=['tracker_position', 'target_position'], 
        title="Tracker and Target"
    )

    # delta tracker target (distances)
    plot_data_time(data_record, key='delta_tracker_target')    

    # plot_genotype_similarity(evo, sim)


def test_plot():
    from dol import simulation    
    sim, data_record = simulation.get_simulation_data_from_random_agent()
    plot_results(None, sim, data_record)


if __name__ == "__main__":
    test_plot()


