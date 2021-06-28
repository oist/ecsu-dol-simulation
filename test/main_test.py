from dol.main import main
import numpy as np

def test_2n_exc():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_neurons', '2', 
        '--popsize', '20', 
        '--max_gen', '10',
        '--exclusive_motors_threshold', '0.1'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [88.39240922352474]
    print('✅ test_2n_exc')

def test_2n_exc_rp3_switch():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_neurons', '2', 
        '--popsize', '20', 
        '--max_gen', '10',        
        '--exclusive_motors_threshold', '0.1',
        '--num_random_pairings', '3',
        '--motor_control_mode', 'SWITCH'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [94.84999999999854]
    print('✅ test_2n_exc_rp3_switch')

def test_3n_exc_rp3_switch():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '5',
        '--num_neurons', '3', 
        '--popsize', '20', 
        '--max_gen', '10',        
        '--exclusive_motors_threshold', '0.1',
        '--num_random_pairings', '3',
        '--motor_control_mode', 'SWITCH'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [94.07571341165567]
    print('✅ test_3n_exc_rp3_switch')

def test_2n_exc_rp3_overlap():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_neurons', '2', 
        '--popsize', '20', 
        '--max_gen', '10',        
        '--num_random_pairings', '3',
        '--motor_control_mode', 'OVERLAP'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [2.0250736246180168]
    print('✅ test_2n_exc_rp3_overlap')

def test_2n_exc_rp3_np2():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_neurons', '2', 
        '--num_pop', '2',
        '--popsize', '20', 
        '--max_gen', '10',        
        '--exclusive_motors_threshold', '0.1',
        '--num_random_pairings', '3',   
        '--motor_control_mode', 'SEPARATE'      
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [94.84999999999854, 94.84999999999854]
    print('✅ test_2n_exc_rp3_np2')

def test_2n_exc_rp3_np4_switch():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_neurons', '2', 
        '--num_pop', '4',
        '--popsize', '20', 
        '--max_gen', '10',        
        '--exclusive_motors_threshold', '0.1',
        '--num_random_pairings', '3', 
        '--motor_control_mode', 'SWITCH'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [90.97311493634516, 93.7696582732533, 93.5070923210551, 93.3720965701068]
    print('✅ test_2n_exc_rp3_np4_switch')

if __name__ == "__main__":
    test_2n_exc()
    test_2n_exc_rp3_switch()
    test_3n_exc_rp3_switch()
    test_2n_exc_rp3_overlap()
    test_2n_exc_rp3_np2()
    test_2n_exc_rp3_np4_switch()