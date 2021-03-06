"""
Rerun a simulation of a given seed and optionally visualize
animation and data plots of behavior and neural activity.
Run as
python -m dol.run_from_dir --help
"""

import os
from numpy.random import RandomState
from dol.simulation import Simulation, MAX_MEAN_DISTANCE
from pyevolver.evolution import Evolution
from dol import utils


def run_simulation_from_dir(dir, generation, genotype_idx=0, population_idx=0, select_sim=1,
                            random_target_seed=None, random_pairing_seed=None, isolation_idx=None,
                            write_data=False, **kwargs):
    """
    Utitity function to get data from a simulation
    """
    sim_index = select_sim - 1

    evo_files = [f for f in os.listdir(dir) if f.startswith('evo_')]
    assert len(evo_files) > 0, "Can't find evo files in dir {}".format(dir)
    file_num_zfill = len(evo_files[0].split('_')[1].split('.')[0])
    generation_str = str(generation).zfill(file_num_zfill)
    sim_json_filepath = os.path.join(dir, 'simulation.json')
    evo_json_filepath = os.path.join(dir, 'evo_{}.json'.format(generation_str))
    sim = Simulation.load_from_file(sim_json_filepath)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)

    data_record_list = []

    random_seed = evo.pop_eval_random_seed

    expect_same_results = True

    # overwriting simulaiton
    if random_target_seed is not None:
        print("Using random target")
        # standard target was initialized in sim.__post_init__
        # so this is going to overwrite it
        sim.init_target(RandomState(random_target_seed))
        expect_same_results = False
    if random_pairing_seed is not None:
        print("Setting random pairing with seed ", random_pairing_seed)
        random_seed = random_pairing_seed
        expect_same_results = False

    original_populations = evo.population_unsorted

    # get the indexes of the populations as they were before being sorted by performance
    # we only need to do this for the first population (index 0)
    original_genotype_idx = evo.population_sorted_indexes[population_idx][genotype_idx]

    performance, sim_perfs, _ = sim.run_simulation(
        original_populations,
        original_genotype_idx,
        random_seed,
        population_idx,
        isolation_idx,
        data_record_list
    )

    performance = sim.normalize_performance(performance)

    if not kwargs.get('quiet', False):
        if genotype_idx == 0:
            perf_orig = evo.best_performances[generation][population_idx]
            perf_orig = sim.normalize_performance(perf_orig)
            print("Performace original: {}".format(perf_orig))
        print("Performace recomputed: {}".format(performance))
        if expect_same_results:
            assert abs(perf_orig - performance) < 1e-5
        # if performance == perf_orig:
        #     print("Exact!!")
        if sim.num_agents == 2:
            print("Sim agents similarity: ", sim.agents_similarity[sim_index])

    if write_data:
        for s, data_record in enumerate(data_record_list, 1):
            if len(data_record_list) > 1:
                outdir = os.path.join(dir, 'data', 'sim_{}'.format(s))
            else:
                outdir = os.path.join(dir, 'data')
            utils.make_dir_if_not_exists_or_replace(outdir)
            for k, v in data_record.items():
                if type(v) is dict:
                    # summary
                    outfile = os.path.join(outdir, '{}.json'.format(k))
                    utils.save_json_numpy_data(v, outfile)
                else:
                    outfile = os.path.join(outdir, '{}.json'.format(k))
                    utils.save_json_numpy_data(v, outfile)

    return performance, sim_perfs, evo, sim, data_record_list


if __name__ == "__main__":
    import argparse
    from dol import plot
    from dol.visual import Visualization
    from dol.visual2d import Visualization2D

    parser = argparse.ArgumentParser(
        description='Rerun simulation'
    )

    # args for run_simulation_from_dir
    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--generation', type=int, help='Number of generation to load')
    parser.add_argument('--genotype_idx', type=int, default=0, help='Index of agent in population to load')
    parser.add_argument('--population_idx', type=int, default=0,
                        help='Index of the population, usually 0 can be 1 in dual populations')
    parser.add_argument('--select_sim', type=int, default=1,
                        help='Which simulation to select for visualization and plot (1-based)')
    parser.add_argument('--random_target_seed', type=int,
                        help='Seed to re-run simulation with random target (None to obtain same results)')
    parser.add_argument('--random_pairing_seed', type=int,
                        help='Seed to re-run simulation with random pairing (None to obtain same results)')
    parser.add_argument('--isolation_idx', type=int,
                        help='To force the first (0) or second (1) agent to run in isolation (None otherwise)')
    parser.add_argument('--write_data', action='store_true', help='Whether to output data (same directory as input)')

    # additional args
    parser.add_argument('--compute_complexity', action='store_true', help='Whether to plot the data')
    parser.add_argument('--visualize_trial', type=int, default=-1, help='Whether to visualize a certain trial')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the data')
    parser.add_argument('--plot_trial', type=int, help='Whether to plot a specif trial')

    args = parser.parse_args()

    perf, sim_perfs, evo, sim, data_record_list = run_simulation_from_dir(**vars(args))

    single_simulation = len(data_record_list) == 1
    data_record = data_record_list[args.select_sim - 1]

    if args.compute_complexity:
        from dol.analyze_complexity import get_sim_agent_complexity
        nc, h = get_sim_agent_complexity (sim_perfs, sim, data_record_list,
            analyze_sensors=True,
            analyze_brain=True,
            analyze_motors=False,
            use_brain_derivatives=False,
            combined_complexity=False,
            rs=RandomState(1))
        print('TSE', nc)
        print('H', h)

    if args.visualize_trial > 0:
        vis = Visualization(sim) if sim.num_dim == 1 else Visualization2D(sim)
        vis.start_simulation_from_data(args.visualize_trial - 1, data_record)
    if args.plot:
        plot.plot_results(evo, sim, args.plot_trial, data_record)

