"""
TODO: Missing module docstring
"""

import os
from numpy.random import RandomState
from dol.simulation import Simulation
from dol import utils


def run_simulation_from_dir(**kwargs):    
    ''' 
    utitity function to get data from a simulation
    '''    
    dir = kwargs['dir']
    generation = kwargs['generation']
    genotype_idx = kwargs['genotype_idx']
    write_data = kwargs['write_data']

    from pyevolver.evolution import Evolution
    evo_files = [f for f in os.listdir(dir) if f.startswith('evo_')]
    assert len(evo_files)>0, "Can't find evo files in dir {}".format(dir)
    file_num_zfill = len(evo_files[0].split('_')[1].split('.')[0])
    generation = str(generation).zfill(file_num_zfill)
    sim_json_filepath = os.path.join(dir, 'simulation.json')
    evo_json_filepath = os.path.join(dir, 'evo_{}.json'.format(generation))
    sim = Simulation.load_from_file(sim_json_filepath)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)

    random_target_seed = kwargs['random_target_seed']
    if random_target_seed is not None:
        print("Setting random target")
        sim.init_target(RandomState(random_target_seed))

    data_record = {}
    # get the indexes of the populations as they were before being sorted by performance
    genotype_idx_unsorted = evo.population_sorted_indexes[genotype_idx]
    random_seed = evo.pop_eval_random_seed
    
    performance = sim.run_simulation(
        evo.population_unsorted, 
        genotype_idx_unsorted, 
        data_record
    )

    print("Performace recomputed: {}".format(performance))

    if write_data:        
        outdir = os.path.join(dir, 'data')
        utils.make_dir_if_not_exists_or_replace(outdir)        
        for k,v in data_record.items():
            if type(v) is dict: 
                # summary
                outfile = os.path.join(outdir, '{}.json'.format(k))
                utils.save_json_numpy_data(v, outfile)
            else:
                outfile = os.path.join(outdir, '{}.json'.format(k))
                utils.save_json_numpy_data(v, outfile)
                        

    return evo, sim, data_record

if __name__ == "__main__":
    import argparse
    from dol import plot
    from dol.visual import Visualization

    parser = argparse.ArgumentParser(
        description='Rerun simulation'
    )

    parser.add_argument('--dir', type=str, help='Directory path')    
    parser.add_argument('--generation', type=int, help='Number of generation to load')
    parser.add_argument('--genotype_idx', type=int, default=0, help='Index of agent in population to load')    
    parser.add_argument('--random_target_seed', type=int, help='Index of agent in population to load')    
    parser.add_argument('--write_data', action='store_true', help='Whether to output data (same directory as input)')
    parser.add_argument('--visualize_trial', type=int, default=-1, help='Whether to visualize a certain trial')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the data')
    
    args = parser.parse_args()
    
    evo, sim, data_record = run_simulation_from_dir(**vars(args))
    
    if args.visualize_trial > 0:
        vis = Visualization(sim)
        vis.start_simulation_from_data(args.visualize_trial-1, data_record)
    if args.plot:
        plot.plot_results(evo, sim, data_record)
