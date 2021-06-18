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
    # print(last_best_perf)
    assert last_best_perf == [4911.746627237504]
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
    # print(last_best_perf)
    assert last_best_perf == [4905.150000000001]
    print('✅ test_2n_exc_rp3_switch')

def test_2n_exc_rp3_dual():
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
    # print(last_best_perf)
    assert last_best_perf == [4908.040022980634, 4906.640307404024]
    print('✅ test_2n_exc_rp3_dual')

if __name__ == "__main__":
    test_2n_exc()
    test_2n_exc_rp3_switch()
    test_2n_exc_rp3_dual()