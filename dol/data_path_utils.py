import os

# lookup tables for simulations in data directory

phil_trans_si_data = os.path.join('data', 'phil_trans_si')

phil_trans_si_data_overlap = {
    'group': os.path.join(
        phil_trans_si_data, 
        '1d_2n_exc-0.1_zfill_rp-3_switch'
    ),
    'joint': os.path.join(
        phil_trans_si_data, 
        '1d_2n_exc-0.1_zfill_rp-0_switch'
    ),
    'individual': os.path.join(
        phil_trans_si_data, 
        '1d_2n_exc-0.1_zfill_rp-3_np-4_switch'
    )
}

phil_trans_si_data_overlap = {
    'group': os.path.join(
        phil_trans_si_data, 
        '1d_2n_zfill_rp-3_overlap'
    ),
    'joint': os.path.join(
        phil_trans_si_data, 
        '1d_2n_zfill_rp-0_overlap'
    ),
    'individual': os.path.join(
        phil_trans_si_data, 
        '1d_2n_zfill_rp-3_np-4_overlap'
    )
}

