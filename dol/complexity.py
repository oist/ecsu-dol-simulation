import numpy as np
import os

def bipartition(collection):
    assert len(collection) > 1
    if len(collection) == 2:
        yield [ [collection[0]], [collection[1]] ]
        return
    first = collection[0]
    rest = collection[1:]
    # put `first` in its own subset 
    yield [ [ first ] ] + [ rest ]        
    for smaller in bipartition(rest):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]    


def sort_bipartition_by_size(bipart_list):
    for bipart in bipart_list:
        bipart.sort(key = len)


def get_entropy_using_coveriance(data):
    assert data.ndim == 2
    num_nodes = len(data)
    cov_matrix = np.cov(data) # num_nodes x num_nodes    
    if num_nodes == 1: # single row
        cov_det = float(cov_matrix) # cov_matrix is a 0-dim array
    else:
        cov_det = np.linalg.det(cov_matrix)
    entropy = 0.5 * np.log( ((2 * np.pi * np.e) ** num_nodes) * cov_det)
    return entropy
    

def test_bipartitions(n):
    l = list(range(1,n+1))
    bipart_list = list(bipartition(l))
    sort_bipartition_by_size(bipart_list) # sort by size of subpartitions
    num = len(bipart_list)
    assert num == (2 ** n - 2) / 2
    print('Bipartition of {} ({}):'.format(l, num))
    print('\n'.join(str(x) for x in bipart_list))

def compute_neural_complexity(data):
    assert data.ndim == 2
    num_nodes, _ = data.shape    
    h_AB = get_entropy_using_coveriance(data)
    # print('entropy AB: ', h_AB)
    index_list = list(range(num_nodes))
    bipart_list = list(bipartition(index_list)) # all possible bipartitions of indexes
    sort_bipartition_by_size(bipart_list) # sort by size of bipartitions sets
    
    # keep track of mi values of each group indexed by k-1
    # (where k is the size of smallest bipartition set)
    mi_group_size = int(num_nodes/2)
    mi_groups_val = np.zeros(mi_group_size) 
    mi_groups_len = np.zeros(mi_group_size) 
    for bipart in bipart_list:
        A = data[bipart[0], :] # rows specified in indexes of first bipartition set
        B = data[bipart[1], :] # rows specified in indexes of second bipartition set
        k = len(A)
        h_A = get_entropy_using_coveriance(A)
        h_B = get_entropy_using_coveriance(B)
        mi = h_A + h_B - h_AB
        mi_groups_val[k-1] += mi
        mi_groups_len[k-1] += 1
    neural_complexity = np.sum(mi_groups_val / mi_groups_len) # sum over averaged values
    return neural_complexity


def test_complexity_random_uniform(num_nodes, num_data_points, seed):
    rs = np.random.RandomState(seed)
    data = rs.uniform(-1,1,size=(num_nodes, num_data_points))
    nc = compute_neural_complexity(data)
    print('Nerual Complexity (Random Uniform)', nc)


def test_complexity_random_gaussian(num_nodes, num_data_points, seed):
    rs = np.random.RandomState(seed)
    data = rs.normal(size=(num_nodes, num_data_points))
    nc = compute_neural_complexity(data)
    print('Nerual Complexity (Random Normal Gaussian)', nc)


def test_complexity_constant(num_nodes, num_data_points, seed):
    rs = np.random.RandomState(seed)
    data = np.ones((num_nodes, num_data_points))
    noise = rs.normal(0, 1e-15, data.shape)
    data = data + noise
    nc = compute_neural_complexity(data)
    print('Nerual Complexity (Constat-ones)', nc)


def generate_correlated_data(num_data_points, cov, rs):
    # see https://quantcorner.wordpress.com/2018/02/09/generation-of-correlated-random-numbers-using-python/
    from scipy.linalg import cholesky
    from scipy.stats import pearsonr
    
    # Compute the (upper) Cholesky decomposition matrix
    corr_mat= np.array(
        [
            [1.0, cov],            
            [cov, 1.0]
        ]
    )

    # Compute the (upper) Cholesky decomposition matrix
    upper_chol = cholesky(corr_mat)

    # Generate 3 series of normally distributed (Gaussian) numbers
    rnd = rs.normal(0.0, 1.0, size=(num_data_points, 2))

    result = rnd @ upper_chol
    corr_0_1 , _ = pearsonr(result[:,0], result[:,1])

    # print(result.shape)
    # print(corr_0_1)
    
    return np.transpose(result)


def test_complexity_correlated_data(num_nodes, num_data_points, cov, seed):
    assert num_nodes%2 == 0

    rs = np.random.RandomState(seed)
    num_pairs = int(num_nodes/2)
    # stuck together pairs of coorelated rows
    data = np.row_stack(
        [
            generate_correlated_data(num_data_points, cov, rs)
            for _ in range(num_pairs)
        ]
    )
    assert data.shape == (num_nodes, num_data_points)
    nc = compute_neural_complexity(data)
    print('Nerual Complexity (Correlated)', nc)

def compute_mutual_information(AB):    
    assert AB.ndim == 2
    A = np.expand_dims(AB[0,:],0) # first row as 2 dim array
    B = np.expand_dims(AB[1,:],0) # second row as 2 dim array
    h_A = get_entropy_using_coveriance(A)
    h_B = get_entropy_using_coveriance(B)
    h_AB = get_entropy_using_coveriance(AB)    
    return h_A + h_B - h_AB


def test_mutual_information(seed):
    num_data_points = 500
    
    rs = np.random.RandomState(seed)    
    mi_random_uniform = compute_mutual_information(
        rs.uniform(-1,1,size=(2, num_data_points))
    )

    rs = np.random.RandomState(seed)    
    mi_random_gaussian = compute_mutual_information(
        rs.normal(size=(2, num_data_points))
    )
    
    rs = np.random.RandomState(seed)    
    mi_random_gaussian = compute_mutual_information(
        rs.normal(size=(2, num_data_points))
    )

    rs = np.random.RandomState(seed)    
    mi_constant = compute_mutual_information(
        np.ones((2, num_data_points)) +
        rs.normal(0, 1e-15, size=(2, num_data_points))
    )

    rs = np.random.RandomState(seed)    
    mi_correlated = compute_mutual_information(
        generate_correlated_data(num_data_points, cov=0.9, rs=rs)
    )
    print('MI random uniform:', mi_random_uniform)
    print('MI random gaussian:', mi_random_gaussian)
    print('MI constant:', mi_constant)
    print('MI correlated:', mi_correlated)


def test_complexity(num_nodes, seed):
    num_data_points=500
    
    test_complexity_random_uniform(num_nodes, num_data_points, seed)
    
    test_complexity_random_gaussian(num_nodes, num_data_points, seed)

    test_complexity_constant(num_nodes, num_data_points, seed)

    test_complexity_correlated_data(num_nodes, num_data_points, cov=0.9, seed=seed)

def test():
    # test_bipartitions(n=4)
    # generate_correlated_data(num_data_points=500, cov=0.9, seed=0, rs = np.random.RandomState())
    # test_mutual_information(seed=0)
    test_complexity(num_nodes=6,seed=0)

def main_single_agent():    
    import matplotlib.pyplot as plt
    from dol.run_from_dir import run_simulation_from_dir    
    from dol.shannon_entropy import get_shannon_entropy_2d

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1,1,1)

    for seed_num in range(1,21):

        if seed_num in [1,7,8,9,14,15]:
            continue

        print('\nseed', seed_num)

        seed_num_zero = str(seed_num).zfill(3)
        dir = './data/2n_exc-0.1/seed_{}'.format(seed_num_zero)
        max_gen = 2000
        step = 500    

        # data elements on which complexity is analyzed
        # trials x 1/2 agents x 500 data points x 2 dim
        data_keys = [
            'agents_sensors', # dim = num sensor = 2
            'agents_brain_output', # dim = num neurons = 2
            'agents_motors' # dim = num motors = 2
        ]    

        num_generations_list = list(range(0,max_gen+step,step)) # 0, 500, ...
        nc_generations = np.zeros(len(num_generations_list))
        h_generations = np.zeros(len(num_generations_list))

        for g, generation in enumerate(num_generations_list):
            print("generation:",generation)
            evo, sim, data_record_list = run_simulation_from_dir(dir, generation)
            data_record = data_record_list[0]   
            data = np.array([ data_record[k] for k in data_keys])
            assert data.shape == (len(data_keys), sim.num_trials, sim.num_agents, sim.num_data_points, 2)
            nc_trials = np.zeros(sim.num_trials)
            h_trials = np.zeros(sim.num_trials)
            for t in range(sim.num_trials):
                # print("trial:",t+1)
                trial_data = data[:,t,:,:,:]
                # trial_data.shape == (3, 1, 500, 2)
                a = 0 # assume only one agent
                trial_data_agent = trial_data[:,a,:,:] 
                # trial_data_agent.shape == (3, 500, 2)
                trial_data_agent = np.moveaxis(trial_data_agent, 2, 1)
                trial_data_agent = trial_data_agent.reshape((6,-1))
                # second_brain_neuron = trial_data_agent[3,:]
                # check = data_record['agents_brain_output'][t][a][:,1]
                # assert (second_brain_neuron==check).all
                brain_trial_data_agent = trial_data_agent[[2,3],:] # only brain
                # trial_data_agent = trial_data_agent[[0,1,2,3],:] # only sensors and brain
                nc = compute_neural_complexity(brain_trial_data_agent) 
                # print("nc:",nc)
                nc_trials[t] = nc
                h_trials[t] = get_shannon_entropy_2d(np.transpose(brain_trial_data_agent))
            nc_avg = np.mean(nc_trials)
            h_avg = np.mean(h_trials)
            print("nc_avg:",nc_avg)
            print("h_avg:",h_avg)
            nc_generations[g] = nc_avg
            h_generations[g] = h_avg
        ax.plot(num_generations_list, nc_generations)
        # ax.plot(num_generations_list, h_generations)
        
        # if nc_generations[-1] > nc_generations[0]:
        #     print('seed --> ', seed_num)

    plt.show()
        
    #     break

    #     assert len(data_record['agents_sensors']) == sim.num_trials        
    #     for t, trial_data in enumerate(data_record):            
    #         assert len(trial_data) == sim.num_agents            
    #         for a in range(sim.num_agents):
    #             agent_data = trial_data[a]
    #             data = np.row_stack(
    #                 [agent_data[k] for k in data_keys]
    #             )
    #             print(data.shape)
    
    # sim_json_filepath = os.path.join(dir, 'simulation.json')
    # sim = Simulation.load_from_file(sim_json_filepath)
    # evo_file_list = [f for f in sorted(os.listdir(dir)) if f.startswith('evo_')]
    # assert len(evo_file_list)>0, "Can't find evo files in dir {}".format(dir)
    # for evo_file in evo_file_list:
    #     gen_num = int(evo_file.split('_')[1].split('.')[0])        
    #     evo_json_filepath = os.path.join(dir, evo_file)
    #     evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)

def main_multi_agent():    
    import matplotlib.pyplot as plt
    from dol.run_from_dir import run_simulation_from_dir    
    from dol.shannon_entropy import get_shannon_entropy_2d

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1,1,1)

    for seed_num in range(1,21):

        if seed_num in [5]:
            continue

        print('\nseed', seed_num)

        seed_num_zero = str(seed_num).zfill(3)
        dir = './data/2n_exc-0.1_rp-3_dual/seed_{}'.format(seed_num_zero)
        max_gen = 2000
        step = 500    

        # data elements on which complexity is analyzed
        # trials x 1/2 agents x 500 data points x 2 dim
        data_keys = [
            'agents_sensors', # dim = num sensor = 2
            'agents_brain_output', # dim = num neurons = 2
            'agents_motors' # dim = num motors = 2
        ]    

        num_generations_list = list(range(0,max_gen+step,step)) # 0, 500, ...
        nc_generations = np.zeros(len(num_generations_list))
        h_generations = np.zeros(len(num_generations_list))

        for g, generation in enumerate(num_generations_list):
            print("generation:",generation)
            evo, sim, data_record_list = run_simulation_from_dir(dir, generation)
            data_record = data_record_list[0]   
            data = np.array([ data_record[k] for k in data_keys])
            assert data.shape == (len(data_keys), sim.num_trials, sim.num_agents, sim.num_data_points, 2)
            nc_trials = np.zeros(sim.num_trials)
            h_trials = np.zeros(sim.num_trials)
            for t in range(sim.num_trials):
                # print("trial:",t+1)
                trial_data = data[:,t,:,:,:]
                # trial_data.shape == (3, 1, 500, 2)
                a = 0 # assume only one agent
                trial_data_agent = trial_data[:,a,:,:] 
                # trial_data_agent.shape == (3, 500, 2)
                trial_data_agent = np.moveaxis(trial_data_agent, 2, 1)
                trial_data_agent = trial_data_agent.reshape((6,-1))
                # second_brain_neuron = trial_data_agent[3,:]
                # check = data_record['agents_brain_output'][t][a][:,1]
                # assert (second_brain_neuron==check).all
                brain_trial_data_agent = trial_data_agent[[2,3],:] # only brain
                # trial_data_agent = trial_data_agent[[0,1,2,3],:] # only sensors and brain
                nc = compute_neural_complexity(brain_trial_data_agent) 
                # print("nc:",nc)
                nc_trials[t] = nc
                h_trials[t] = get_shannon_entropy_2d(np.transpose(brain_trial_data_agent))
            nc_avg = np.mean(nc_trials)
            h_avg = np.mean(h_trials)
            print("nc_avg:",nc_avg)
            print("h_avg:",h_avg)
            nc_generations[g] = nc_avg
            h_generations[g] = h_avg
        ax.plot(num_generations_list, nc_generations)
        # ax.plot(num_generations_list, h_generations)
        
        # if nc_generations[-1] > nc_generations[0]:
        #     print('seed --> ', seed_num)

    plt.show()
        
    #     break

    #     assert len(data_record['agents_sensors']) == sim.num_trials        
    #     for t, trial_data in enumerate(data_record):            
    #         assert len(trial_data) == sim.num_agents            
    #         for a in range(sim.num_agents):
    #             agent_data = trial_data[a]
    #             data = np.row_stack(
    #                 [agent_data[k] for k in data_keys]
    #             )
    #             print(data.shape)
    
    # sim_json_filepath = os.path.join(dir, 'simulation.json')
    # sim = Simulation.load_from_file(sim_json_filepath)
    # evo_file_list = [f for f in sorted(os.listdir(dir)) if f.startswith('evo_')]
    # assert len(evo_file_list)>0, "Can't find evo files in dir {}".format(dir)
    # for evo_file in evo_file_list:
    #     gen_num = int(evo_file.split('_')[1].split('.')[0])        
    #     evo_json_filepath = os.path.join(dir, evo_file)
    #     evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)


if __name__ == "__main__":    
    # main_single_agent()
    main_multi_agent()
    
    
    
    
    