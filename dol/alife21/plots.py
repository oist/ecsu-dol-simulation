import numpy as np
from dol.run_from_dir import run_simulation_from_dir
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_activity_over_time(data_record, key, trial, title, filename, save_fig=True, joint=False):
    """
    Line plot of simulation run for a specific key over simulation time stpdf.
    """
    trial_data = data_record[key][trial]
    fig = plt.figure(figsize=(5, 3))
    fig.suptitle(title, fontsize=16)

    if key == 'agents_brain_output':
        node_label = 'N'
        ylims = [0, 1]
    elif key == 'agents_motors':
        node_label = 'M'
        ylims = [0, 7]
    else:
        node_label = 'data'
        ylims = [np.floor(np.amin(trial_data)), np.ceil(np.amax(trial_data))]

    if joint:
        num_agents = 2
        for a in range(num_agents):
            ax = fig.add_subplot(num_agents, 1, a + 1)
            agent_trial_data = trial_data[a]
            for n in range(agent_trial_data.shape[1]):
                ax.plot(agent_trial_data[:, n], label='{}{}'.format(node_label, n + 1))
                if key == 'agents_motors':
                    ax.axhline(0.1, color='grey', linestyle='--')
                handles, labels = ax.get_legend_handles_labels()
                ax.set_title('Agent {}'.format(a+1))
                ax.set_ylim(ylims)
        ax.legend(handles, labels, loc='upper right',
                  fontsize="medium", markerscale=0.5, labelspacing=0.1)
    else:
        ax = fig.add_subplot(1, 1, 1)
        for n in range(trial_data[0].shape[1]):
            ax.plot(trial_data[0][:, n], label='{}{}'.format(node_label, n + 1))
            if key == 'agents_motors':
                ax.axhline(0.1, color='grey', linestyle='--')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper right',
                      fontsize="medium", markerscale=0.5, labelspacing=0.1)
            ax.set_ylim(ylims)
    sns.despine()
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
    else:
        plt.show()


def plot_data_time_multi_keys(data_record, trial, title, filename, save_fig=True):
    """
    Plot several keys in the same plot, e.g. both target and tracker position.
    """
    fig = plt.figure(figsize=(10, 3))
    fig.suptitle(title, fontsize=16)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data_record['tracker_position'][trial], label='Tracker position')
    ax.plot(data_record['target_position'][trial], label='Target position')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right',
              fontsize="medium", markerscale=0.5, labelspacing=0.1)
    sns.despine()
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
    else:
        plt.show()


def get_derivative(network, state):
    # Compute the next state of the network given its current state and the simple euler equation
    # update the state of all neurons
    # let n the number of neurons
    # output shape is (n, ): [O1, O2, ..., ON]
    # weights shape is (n, n): [[W11,W12,...,W1n],[W21,W22,...,W2n],...,[Wn1,Wn2,...,Wnn]]
    # np.dot(outputs, weights) returns a vector of shape (n,): (n,)·(n,n) = (n,)
    # [O1·W11 + O2·W21 + ... + On·Wn1, O1·W12 + O2·W22 + ... + On·Wn2, ..., O1·W1n + O2·W2n + ... + On·Wnn]
    dy_dt = \
        network.step_size * \
        np.multiply(
            1 / network.taus,
            - state + np.dot(network.output, network.weights) + network.input
        )
    return dy_dt


def plot_phase_space(net_history, network):
    """ Plot the phase portrait
    We'll use matplotlib quiver function, which wants as arguments the grid of x and y coordinates,
    and the derivatives of these coordinates.
    In the plot we see the locations of stable and unstable equilibria,
    and can eyeball the trajectories that the system will take through
    the state space by following the arrows.
    """
    # Define the sample space (plotting ranges)
    # ymin = np.amin(net_history)
    # ymax = np.amax(net_history)
    ymin = -10
    ymax = 10
    y1 = np.linspace(ymin, ymax, 30)
    y2 = np.linspace(ymin, ymax, 30)
    Y1, Y2 = np.meshgrid(y1, y2)
    dim_y = y1.shape[0]

    # calculate the state space derivatives across our sample space
    changes_y1 = np.zeros([dim_y, dim_y])
    changes_y2 = np.zeros([dim_y, dim_y])

    for i in range(dim_y):
        for j in range(dim_y):
            changes = get_derivative(network, np.array([Y1[i, j], Y2[i, j]]))
            changes_y1[i,j] = changes[0]
            changes_y2[i,j] = changes[1]

    plt.figure(figsize=(10,6))
    plt.quiver(Y1, Y2, changes_y1, changes_y2, color='b', alpha=.75)
    plt.plot(net_history[:, 0], net_history[:, 1], color='r')
    plt.scatter(net_history[0][0], net_history[0][1], color='orange', zorder=1)
    plt.xlabel('y1', fontsize=14)
    plt.ylabel('y2', fontsize=14)
    plt.title('Phase portrait and a single trajectory for agent brain', fontsize=16)
    plt.show()

def plot_alife21(indir, outdir):
    # run simulation with random target
    # returns average performance normalized, a list of performances per trial,
    # evolution object, simulation object, simulation data
    # individual
    plot_seed = '017'
    plot_dir = f'{indir}/1d_2n_exc-0.1_zfill/seed_{plot_seed}'

    performance1, sim_perfs1, evo1, sim1, data_record_list1, sim_idx = run_simulation_from_dir(
        dir=plot_dir, generation=5000, random_target_seed=78)

    data_record1 = data_record_list1[0]
    trl = 0

    plot_data_time_multi_keys(
        data_record1, 
        trl,
        "Tracker and Target", 
        os.path.join(outdir, 'individual_behavior.pdf')
    )
    plot_activity_over_time(
        data_record1, 
        'agents_brain_output', 
        trl,
        "Brain output", 
        os.path.join(outdir, 'individual_brain.pdf')
    )
    plot_activity_over_time(
        data_record1, 
        'agents_motors', 
        trl,
        "Motor output", 
        os.path.join(outdir, 'individual_motor.pdf')
    )

    # plot_phase_space
    # sim1_trajectory = data_record1['agents_brain_state'][0][0] # first trial, first agent
    # sim1_brain = sim1.agents[0].brain # first agent brain network
    # plot_phase_space(sim1_trajectory, sim1_brain)

    # generalists
    plot_seed = '019'
    plot_dir = f'{indir}/1d_2n_exc-0.1_zfill_rp-3_switch/seed_{plot_seed}'
    _, sim_perfs2, _, _, _, _ = run_simulation_from_dir(
        dir=plot_dir, generation=5000)
    # select best simulation for joint cases
    sim_idx = np.argmax(sim_perfs2)

    # rerun with random target
    performance2, sim_perfs2, evo2, sim2, data_record_list2, sim_idx2 = run_simulation_from_dir(
        dir=plot_dir, generation=5000, random_target_seed=78)
    data_record2 = data_record_list2[sim_idx]

    plot_data_time_multi_keys(
        data_record2, 
        trl,
        "Tracker and Target", 
        os.path.join(outdir, 'generalist_behavior.pdf')
    )
    plot_activity_over_time(
        data_record2, 
        'agents_brain_output', 
        trl,
        "Brain output", 
        os.path.join(outdir, 'generalist_brain.pdf'), 
        joint=True
    )
    plot_activity_over_time(
        data_record2, 
        'agents_motors', 
        trl,
        "Motor output", 
        os.path.join(outdir, 'generalist_motor.pdf'), 
        joint=True
    )

    # plot_phase_space
    sim2_trajectory1 = data_record2['agents_brain_state'][0][0]
    sim2_brain1 = sim2.agents[0].brain
    plot_phase_space(sim2_trajectory1, sim2_brain1)

    # specialists
    plot_seed = '003'
    plot_dir = f'{indir}/1d_2n_exc-0.1_zfill_rp-3_dual/seed_{plot_seed}'

    _, sim_perfs3, _, _, _, _ = run_simulation_from_dir(
        dir=plot_dir, generation=5000)
    # select best simulation for joint cases
    sim_idx = np.argmax(sim_perfs3)

    # rerun with random target
    performance3, sim_perfs3, evo3, sim3, data_record_list3, sim_idx3 = run_simulation_from_dir(
        dir=plot_dir, generation=5000, random_target_seed=78)

    data_record3 = data_record_list3[sim_idx]


    plot_data_time_multi_keys(
        data_record3, trl,
        "Tracker and Target", 
        os.path.join(outdir, 'specialist_behavior.pdf')
    )
    plot_activity_over_time(
        data_record3, 
        'agents_brain_output', 
        trl,
        "Brain output", 
        os.path.join(outdir, 'specialist_brain.pdf'), 
        joint=True
    )
    plot_activity_over_time(
        data_record3, 
        'agents_motors', 
        trl,
        "Motor output", 
        os.path.join(outdir, 'specialist_motor.pdf'), 
        joint=True
    )
