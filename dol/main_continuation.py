"""
TODO: Missing module docstring
"""

import os
from dol import gen_structure
from dol.simulation import Simulation
from dol import utils
from pyevolver.evolution import Evolution
import numpy as np

import argparse
from pytictoc import TicToc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run the Division of Labor Simulation (continuation)'
    )

    parser.add_argument('--dir', type=str, help='Output directory')
    parser.add_argument('--num_gen', type=int, help='Number of generations')    
    parser.add_argument('--cores', type=int, default=-1, help='Number of cores')        

    args = parser.parse_args()

    dir = args.dir
    evo_files = sorted([f for f in os.listdir(dir) if f.startswith('evo_')])
    assert len(evo_files)>0, "Can't find evo files in dir {}".format(dir)    
    last_generation = evo_files[-1].split('_')[1].split('.')[0]
    file_num_zfill = len(last_generation)
    sim_json_filepath = os.path.join(dir, 'simulation.json')
    evo_json_filepath = os.path.join(dir, 'evo_{}.json'.format(last_generation))

    assert args.num_gen > int(last_generation), \
        "num_gen is <= of the last available generation ({})".format(last_generation)
        
    sim = Simulation.load_from_file(
        sim_json_filepath
    )
    
    if args.cores>0:
        sim.cores = args.cores
    
    evo = Evolution.load_from_file(
        evo_json_filepath,
        evaluation_function=sim.evaluate,
        max_generation=args.num_gen
    )

    t = TicToc()
    t.tic()

    evo.run()

    print('Ellapsed time: {}'.format(t.tocvalue()))
