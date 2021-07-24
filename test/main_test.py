from dol.main import main
import numpy as np

def test_1d_1n_exc():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_neurons', '1', 
        '--popsize', '20', 
        '--max_gen', '10',
        '--exclusive_motors_threshold', '0.1'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [93.94248760669507]
    print('✅ test_1d_1n_exc')

def test_1d_2n_exc():
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
    print('✅ test_1d_2n_exc')

def test_1d_2n_exc_rp0_switch():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_neurons', '2', 
        '--popsize', '20', 
        '--max_gen', '10',        
        '--exclusive_motors_threshold', '0.1',
        '--num_random_pairings', '0',
        '--motor_control_mode', 'SWITCH'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [88.74146488146835]
    print('✅ test_1d_2n_exc_rp0_switch')

def test_1d_2n_exc_rp3_switch():
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
    print('✅ test_1d_2n_exc_rp3_switch')

def test_1d_3n_exc_rp3_switch():
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
    print('✅ test_1d_3n_exc_rp3_switch')

def test_1d_2n_exc_rp3_overlap():
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
    print('✅ test_1d_2n_exc_rp3_overlap')

def test_1d_2n_exc_rp3_np2():
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
    print('✅ test_1d_2n_exc_rp3_np2')

def test_1d_2n_exc_rp3_np4_switch():
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
    print('✅ test_1d_2n_exc_rp3_np4_switch')

def test_1d():
    test_1d_1n_exc() # isolated
    test_1d_2n_exc() # isolated
    test_1d_2n_exc_rp0_switch()
    test_1d_2n_exc_rp3_switch()
    test_1d_3n_exc_rp3_switch()
    test_1d_2n_exc_rp3_overlap()
    test_1d_2n_exc_rp3_np2()
    test_1d_2n_exc_rp3_np4_switch()

def test_2d_4n():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_dim', '2',
        '--num_neurons', '4', 
        '--popsize', '20', 
        '--max_gen', '10',
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [203.85904344564187]
    print('✅ test_2d_4n')

def test_2d_4n_rp0_switch():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_dim', '2',
        '--num_neurons', '4', 
        '--popsize', '20', 
        '--max_gen', '10',
        '--num_random_pairings', '0', 
        '--motor_control_mode', 'SWITCH'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [209.80775083182743]
    print('✅ test_2d_4n_rp0_switch')

# def test_2d_4n_group():
#     sim, evo = main([             
#         '--cores', '1', 
#         '--seed', '1',
#         '--gen_zfill',
#         '--num_dim', '2',
#         '--num_neurons', '4', 
#         '--popsize', '48', 
#         '--max_gen', '20',
#         '--num_random_pairings', '3', 
#         '--motor_control_mode', 'SWITCH'
#     ])
#     last_best_perf = evo.best_performances[-1]
#     print('✅ test_2d_4n_group')

def test_2d_4n_rp3_switch():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_dim', '2',
        '--num_neurons', '4', 
        '--popsize', '20', 
        '--max_gen', '10',
        '--num_random_pairings', '3', 
        '--motor_control_mode', 'SWITCH'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [301.2480704684185]
    print('✅ test_2d_4n_rp3_switch')

def test_2d_4n_rp3_np4_switch():
    sim, evo = main([             
        # '--dir', './data/tmp', 
        '--cores', '7', 
        '--seed', '1',
        '--num_dim', '2',
        '--num_neurons', '4', 
        '--num_pop', '4',
        '--popsize', '20', 
        '--max_gen', '10',
        '--num_random_pairings', '3', 
        '--motor_control_mode', 'SWITCH'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [206.7847936047856, 207.61790413771269, 207.51109872097186, 209.16469040317497]
    print('✅ test_2d_4n_rp3_np4_switch')

def test_2d(): 
    test_2d_4n() # isolated
    test_2d_4n_rp0_switch()
    test_2d_4n_rp3_switch()
    test_2d_4n_rp3_np4_switch()

if __name__ == "__main__":
    test_1d()
    test_2d()