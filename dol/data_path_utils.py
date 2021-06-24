import os

# lookup tables for simulations in data directory

phil_trans_si_data = os.path.join('data', 'phil_trans_si')

phil_trans_si_data_exclusive_switch = {
    'group': os.path.join( # 8 seeds converged: [1, 5, 8, 9, 12, 14, 15, 19]
        phil_trans_si_data, 
        '1d_2n_exc-0.1_zfill_rp-3_switch'
    ),
    'joint': os.path.join(
        phil_trans_si_data, # 1 seed converged: [6]
        '1d_2n_exc-0.1_zfill_rp-0_switch'
    ),
    'individual': os.path.join(
        phil_trans_si_data, # 0 seeds converged: []
        '1d_2n_exc-0.1_zfill_rp-3_np-4_switch'
    )
}

phil_trans_si_data_overlap = {
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

