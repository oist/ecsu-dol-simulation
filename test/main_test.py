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
    assert normalized_perf == [88.25337276249593]
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
        '--switch_agents_motor_control', 'True'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [94.84999999999854]
    print('✅ test_2n_exc_rp3_switch')

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
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [91.95997701936722, 93.35969259597732]
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
        '--switch_agents_motor_control', 'True'
    ])
    last_best_perf = evo.best_performances[-1]
    normalized_perf = [sim.normalize_performance(x) for x in last_best_perf]
    # print(normalized_perf)
    assert normalized_perf == [89.21638501026064, 92.76933384185577, 92.83506417537319, 92.87406656749044]
    print('✅ test_2n_exc_rp3_np4_switch')

if __name__ == "__main__":
    test_2n_exc()
    test_2n_exc_rp3_switch()
    test_2n_exc_rp3_np2()
    test_2n_exc_rp3_np4_switch()