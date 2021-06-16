from dol.run_from_dir import run_simulation_from_dir

def demo():
    seed = 1
    generation = 5000
    dir = f'data/phil_trans_si/1d_2n_exc-0.1_zfill_rp-3_switch/seed_{str(seed).zfill(3)}'
    perf, sim_perfs, evo, sim, data_record_list, sim_idx = run_simulation_from_dir(
        dir=dir, generation=generation)

if __name__ == "__main__":
    demo()