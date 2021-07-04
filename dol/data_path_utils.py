import os

# lookup tables for simulations in data directory

phil_trans_si_data = os.path.join('data', 'phil_trans_si')

phil_trans_si_data_2n_exclusive_switch = {
    'group': os.path.join( # converged seeds (8): [1, 5, 8, 9, 12, 14, 15, 19]
        phil_trans_si_data, 
        '1d_2n_exc-0.1_zfill_rp-3_switch'
    ),
    'joint': os.path.join(
        phil_trans_si_data, # converged seeds (1): [6]
        '1d_2n_exc-0.1_zfill_rp-0_switch'
    ),
    'individual': os.path.join(
        phil_trans_si_data, # converged seeds (0): []
        '1d_2n_exc-0.1_zfill_rp-3_np-4_switch'
    )
}

phil_trans_si_data_3n_exclusive_switch = {
    'group': os.path.join( # converged seeds (10): [5, 8, 9, 10, 11, 14, 16, 17, 18, 20]
        phil_trans_si_data, 
        '1d_3n_exc-0.1_zfill_rp-3_switch'
    ),
    'joint': os.path.join(
        phil_trans_si_data, # converged seeds (2): [2, 3]
        '1d_3n_exc-0.1_zfill_rp-0_switch'
    ),
    'individual': os.path.join(
        phil_trans_si_data, # converged seeds (1): [13]
        '1d_3n_exc-0.1_zfill_rp-3_np-4_switch'
    )
}

phil_trans_si_data_4n_exclusive_switch = {
    'group': os.path.join( # converged seeds (15): [1, 3, 4, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20]
        phil_trans_si_data, 
        '1d_4n_exc-0.1_zfill_rp-3_switch'
    ),
    'joint': os.path.join(
        phil_trans_si_data, # converged seeds (2): [12, 15]
        '1d_4n_exc-0.1_zfill_rp-0_switch'
    ),
    'individual': os.path.join(
        phil_trans_si_data, # converged seeds (5): [6, 7, 10, 12, 16]
        '1d_4n_exc-0.1_zfill_rp-3_np-4_switch'
    )
}

phil_trans_si_data_10n_exclusive_switch = {
    'group': os.path.join( 
        phil_trans_si_data, 
        '1d_10n_exc-0.1_zfill_rp-3_switch'
    ),
    'joint': os.path.join(
        phil_trans_si_data, 
        '1d_10n_exc-0.1_zfill_rp-0_switch'
    ),
    'individual': os.path.join(
        phil_trans_si_data, 
        '1d_10n_exc-0.1_zfill_rp-3_np-4_switch'
    )
}

phil_trans_si_data_2n_overlap = {
    'group': os.path.join( # 20/20 seeds converged
        phil_trans_si_data, 
        '1d_2n_zfill_rp-3_overlap'
    ),
    'joint': os.path.join( # 20/20 seeds converged
        phil_trans_si_data, 
        '1d_2n_zfill_rp-0_overlap'
    ),
    'individual': os.path.join( # 20/20 seeds converged
        phil_trans_si_data, 
        '1d_2n_zfill_rp-3_np-4_overlap'
    )
}

if __name__ == "__main__":
    from dol.analyze_results import get_last_performance_seeds
    for name, exp_dict in [
        ('2n (overlap)', phil_trans_si_data_2n_overlap),
        ('2n (switch + exclusive motors)', phil_trans_si_data_2n_exclusive_switch),        
        ('3n (switch + exclusive motors)', phil_trans_si_data_3n_exclusive_switch),
        ('4n (switch + exclusive motors)', phil_trans_si_data_4n_exclusive_switch),
        ('10n (switch + exclusive motors)', phil_trans_si_data_10n_exclusive_switch)]:
        
        print('————————————————————')
        print(name.upper())        
        print('————————————————————')
        for type, path in exp_dict.items():
            print(f'--> {type.upper()} ({os.path.basename(path)})')
            converged_seeds = get_last_performance_seeds(path, print_stats=True)
            print()
        
        
