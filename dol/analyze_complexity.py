
num_sensors = 2
num_motors = 2
num_neurons = 4
num_agents = 1
num_data_points = 500
num_trials = 4

data_record = {
    'agents_sensors': [
        [
            [ 
                ['t{}_a{}_p{}_sens{}'.format(t,a,p,s) for s in range(num_sensors)] 
                for p in range(num_data_points)    
            ] 
            for a in range(num_agents)            
        ] 
        for t in range(num_trials)
    ],
    'agents_brain_output': [
        [
            [ 
                ['t{}_a{}_p{}_brain{}'.format(t,a,p,n) for n in range(num_neurons)] 
                for p in range(num_data_points)
            ] 
            for a in range(num_agents)            
        ] 
        for t in range(num_trials)
    ],
    'agents_motors': [
        [
            [ 
                ['t{}_a{}_p{}_mot{}'.format(t,a,p,m) for m in range(num_motors)]                 
                for p in range(num_data_points)
            ] 
            for a in range(num_agents)            
        ] 
        for t in range(num_trials)
    ],
}

def main_single_agent(only_brain=False):    
    import matplotlib.pyplot as plt
    from dol.run_from_dir import run_simulation_from_dir    
    from dol.shannon_entropy import get_shannon_entropy_2d, get_shannon_entropy_dd, get_shannon_entropy_dd_simplified

    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(1,1,1)

    for seed_num in range(1,2): # ,21

        # if seed_num in [1,7,8,9,14,15]:
        # if seed_num in [5, 10, 12, 13, 19]:
        #     continue

        # ax = fig.add_subplot(4, 5, seed_num) # projection='3d'

        print('\nseed', seed_num)

        seed_num_zero = str(seed_num).zfill(3)
        # dir = './data/2n_exc-0.1/seed_{}'.format(seed_num_zero)
        dir = './data/4n_exc-0.1/seed_{}'.format(seed_num_zero)
        max_gen = 2000
        # step = 500    
        step = 200    

        if only_brain:
            data_keys = [
                'agents_brain_output' # dim = num neurons = 2/4
            ]
        else:
            # data elements on which complexity is analyzed
            # trials x 1/2 agents x 500 data points x 2 dim        
            data_keys = [
                'agents_sensors', # dim = num sensor = 2
                'agents_brain_output', # dim = num neurons = 2/4
                'agents_motors' # dim = num motors = 2
            ]    

        num_generations_list = list(range(0,max_gen+step,step)) # 0, 500, ...
        nc_generations = np.zeros(len(num_generations_list))
        h_generations = np.zeros(len(num_generations_list))

        for g, generation in enumerate(num_generations_list):
            print("generation:",generation)
            # evo, sim, data_record_list = run_simulation_from_dir(dir, generation)
            # num_trials = sim.num_trials
            # num_agents = sim.num_agents
            # num_data_points = sim.num_data_points
            # data_record = data_record_list[0]   

            # (num_trials, num_data_points, num_agents, num_neurons) -> 
            # (num_neurons, num_trials, num_data_points, num_agents)
            
            data = [ 
                np.moveaxis(np.array(data_record[k]), 3, 0) # moving last dim (num_sensors/num_neurons/num_mot) first
                for k in data_keys
            ]                
            if only_brain:
                num_rows = num_neurons = len(data[0])                
            else:
                num_sensors, num_neurons, num_motors = [len(d) for d in data]
                num_rows = num_sensors + num_neurons + num_motors        
            # assert sim.num_brain_neurons == num_sensors            
            data = np.stack([r for d in data for r in d ]) # stacking all rows together            
            assert data.shape == (
                num_rows, 
                num_trials, 
                num_agents,
                num_data_points,                
            )
            
            nc_trials = np.zeros(num_trials)
            h_trials = np.zeros(num_trials)
            for t in range(num_trials):
                print("trial:",t+1)
                trial_data = data[:,t,:,:]
                assert trial_data.shape == (num_rows, num_data_points, num_agents)
                a = 0 # assume only one agent
                trial_data_agent = trial_data[:,:,a] 
                assert trial_data_agent.shape == (num_rows, num_data_points)                                
                print(trial_data_agent.shape)
                print(trial_data_agent)
                nc = compute_neural_complexity(trial_data_agent) 
                # print("nc:",nc)
                nc_trials[t] = nc
                h_trials[t] = get_shannon_entropy_dd_simplified(np.transpose(trial_data_agent))
            nc_avg = np.mean(nc_trials)
            h_avg = np.mean(h_trials)
            print("nc_avg:",nc_avg)
            print("h_avg:",h_avg)
            nc_generations[g] = nc_avg
            h_generations[g] = h_avg
        # ax.plot(num_generations_list, nc_generations, label=str(seed_num))
        # ax.plot(num_generations_list, h_generations)
        
        # if nc_generations[-1] > nc_generations[0]:
        #     print('seed --> ', seed_num)

    # plt.legend()
    # plt.show()
        
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
    main_single_agent(only_brain=False)
    # main_multi_agent()
    
    
    
    
    