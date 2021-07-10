import os

# lookup tables for simulations in data directory

overlap_dir = os.path.join('data', 'overlap')
exc_switch_dir = os.path.join('data', 'exc_switch')


exc_switch_xN_dir = lambda x: {
    'group': os.path.join( 
        exc_switch_dir, 
        f'1d_{x}n_exc-0.1_zfill_rp-3_switch'
    ),
    'joint': os.path.join(
        exc_switch_dir,
        f'1d_{x}n_exc-0.1_zfill_rp-0_switch'
    ),
    'individual': os.path.join(
        exc_switch_dir,
        f'1d_{x}n_exc-0.1_zfill_rp-3_np-4_switch'
    )
}

overlap_dir_xN = lambda x: {
    'group': os.path.join( 
        overlap_dir, 
        f'1d_{x}n_zfill_rp-3_overlap'
    ),
    'joint': os.path.join( 
        overlap_dir, 
        f'1d_{x}n_zfill_rp-0_overlap'
    ),
    'individual': os.path.join( 
        overlap_dir, 
        f'1d_{x}n_zfill_rp-3_np-4_overlap'
    )
}

if __name__ == "__main__":
    from dol.analyze_results import get_last_performance_seeds
    for name, exp_dict in [
        ('1n (overlap)', overlap_dir_xN(1)),
        ('2n (overlap)', overlap_dir_xN(2)),
        ('1n (switch + exclusive motors)', exc_switch_xN_dir(1)),        
        ('2n (switch + exclusive motors)', exc_switch_xN_dir(2)),        
        ('3n (switch + exclusive motors)', exc_switch_xN_dir(3)),
        ('4n (switch + exclusive motors)', exc_switch_xN_dir(4)),
        ('5n (switch + exclusive motors)', exc_switch_xN_dir(5)),
        ('10n (switch + exclusive motors)', exc_switch_xN_dir(10))]:
        
        print('————————————————————')
        print(name.upper())        
        print('————————————————————')
        for type, path in exp_dict.items():
            print(f'--> {type.upper()} ({os.path.basename(path)})')
            converged_seeds = get_last_performance_seeds(path, print_stats=True) # compute_nfn=True
            print()
        
        
