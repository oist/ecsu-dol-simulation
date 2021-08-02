import os
from dol import utils

# lookup tables for simulations in data directory
# if run locally, it assumes data is store in project directory 
# to synch data frolder from deigo run
# rsync -avz --include="*/" --include="simulation.json" --include="evo_5000.json" --exclude="*" deigo-ext:'/bucket/FroeseU/fede/dol-simulation/*' ./data
data_path = '/bucket/FroeseU/fede/dol-simulation' if  utils.am_i_on_deigo() else 'data'

exc_switch_dir = os.path.join(data_path, 'exc_switch')
overlap_dir = os.path.join(data_path, 'overlap')

exc_switch_xN_dir = lambda x: {
    'individual': os.path.join(
        exc_switch_dir,
        f'1d_{x}n_exc-0.1_zfill_rp-3_np-2_switch'
    ),
    'group': os.path.join( 
        exc_switch_dir, 
        f'1d_{x}n_exc-0.1_zfill_rp-3_switch'
    ),
    'joint': os.path.join(
        exc_switch_dir,
        f'1d_{x}n_exc-0.1_zfill_rp-1_np-2_noshuffle_switch'
    )

}

overlap_dir_xN = lambda x: {
    'individual': os.path.join( 
        overlap_dir, 
        f'1d_{x}n_zfill_rp-3_np-4_overlap'
    ),
    'group': os.path.join( 
        overlap_dir, 
        f'1d_{x}n_zfill_rp-3_overlap'
    ),
    'joint': os.path.join( 
        overlap_dir, 
        f'1d_{x}n_zfill_rp-0_overlap'
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
            converged_seeds = get_last_performance_seeds(path, print_stats=True) # best_sim_stats='converged'
            print()
        
        
