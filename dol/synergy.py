#! /usr/bin/env python 3

import sys
from dol.run_from_dir import run_simulation_from_dir

def demo():
	acceptedSeeds = [2, 3, 6, 10, 11, 12, 13, 14, 15, 16, 18, 19]
	generation = 5000
	for seed in acceptedSeeds:
		print('++++++++++   ', seed)
	    # seed = 2	    
	    dir = f'data/phil_trans_si/1d_2n_exc-0.1_zfill_rp-3_switch/seed_{str(seed).zfill(3)}'
	    perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(dir = dir, generation = generation)
	    print(type(data_record_list))
	    sys.exit()

if __name__ == "__main__":
    demo()