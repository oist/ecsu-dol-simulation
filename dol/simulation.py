"""
TODO: Missing module docstring
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Tuple, List
import json
import numpy as np
from numpy.random import RandomState
from joblib import Parallel, delayed
from pyevolver.json_numpy import NumpyListJsonEncoder
from pyevolver.timing import Timing
from dol.agent import Agent
from dol.tracker import Tracker
from dol.target import Target
from dol.target2d import Target2D
from dol.tracker2d import Tracker2D
from dol import gen_structure
from dol import utils


ENV_SIZE = 400

# max mean distance from target with v=-2 is 5009.8 
# (see target.test_max_distance)
MAX_MEAN_DISTANCE = 5000

@dataclass
class Simulation:   

    # the genotype structure  
    genotype_structure: Dict = field(default_factory = \
        lambda:gen_structure.DEFAULT_GEN_STRUCTURE(2))  

    num_dim: int = 1 # number of dimensions (1 or 2)

    # random pairings
    num_random_pairings: int = None
    # None -> agents are alone in the simulation (default)
    # 0    -> agents are evolved in pairs: a genotype contains a pair of agents
    # N>0  -> each agent will go though a simulation with N other agents

    switch_agents_motor_control: bool = False  
    # when num_agents is 2 this decides whether the two agents switch control of L/R motors 
    # in different trials (mix=True) or not (mix=False) in which case the first agent
    # always control the left motor and the second the right

    exclusive_motors_threshold:float = None

    dual_population:float = False
    
    brain_step_size: float = 0.1
    num_trials: int = 4
    trial_duration: int = 50    
    num_cores: int = 1
    random_seed: int = 0
    timeit: bool = False    

    def __post_init__(self):    

        self.__check_params__()   

        self.num_agents = 1 if self.num_random_pairings is None else 2 

        self.num_brain_neurons = gen_structure.get_num_brain_neurons(self.genotype_structure)
        self.num_data_points = int(self.trial_duration / self.brain_step_size)

        self.agents = [
            Agent(
                self.num_brain_neurons,
                self.brain_step_size,
                self.genotype_structure,
            )
            for _ in range(self.num_agents)
        ]           
        
        self.init_tracker()
        
        self.init_target()

        self.timing = Timing(self.timeit)                

    def __check_params__(self):
        assert self.num_random_pairings is None or self.num_random_pairings>=0, \
            "num_random_pairings must be None or >= 0"
        
        assert not self.dual_population or self.num_random_pairings > 0, \
            "In dual position num_random_pairings must be > 0"

        assert self.exclusive_motors_threshold is None or self.num_dim==1, \
            "In 2d mode exclusive_motors mode is not appropriate"

    def split_population(self):
        # when population will be split in two for computing random pairs matching
        return not self.dual_population and \
            self.num_random_pairings is not None and \
            self.num_random_pairings>0

    def init_tracker(self):
        if self.num_dim == 1:
            self.tracker = Tracker()
        else:
            assert self.num_dim == 2
            self.tracker = Tracker2D()


    def init_target(self, rs=None):
        if self.num_dim == 1:
            if rs is None:
                # vel_list = np.arange(1, self.num_trials+1) # 1, 2, 3, 4
                vel_list = np.repeat(np.arange(1,self.num_trials/2+1),2)[:self.num_trials] # 1, 1, 2, 2
                vel_list[1::2] *= -1 # +, -, +, -, ...
                self.target = Target(
                    num_data_points = self.num_data_points,
                    trial_vel = vel_list, 
                    trial_start_pos = [0] * self.num_trials,
                    trial_delta_bnd = [0] * self.num_trials
                )
            else:
                # random target
                max_vel = np.ceil(self.num_trials/2)
                max_pos = ENV_SIZE/8
                self.target = Target(
                    num_data_points = self.num_data_points,
                    trial_vel = rs.choice([-1,1]) * rs.uniform(1, max_vel, self.num_trials),
                    trial_start_pos = rs.uniform(-max_pos, max_pos, self.num_trials),
                    trial_delta_bnd = rs.uniform(0, max_pos, self.num_trials)
                )
        else:
            assert self.num_dim == 2
            self.target = Target2D(
                num_data_points = self.num_data_points,
                trial_vel = [1] * self.num_trials,
                trial_start_pos = [np.zeros(2)] * self.num_trials,
            )




    def get_genotypes_similarity(self):
        if self.num_agents == 1:
            return None
        return 1 - np.linalg.norm(np.subtract(self.genotypes[0],self.genotypes[1]))

    def save_to_file(self, file_path):
        with open(file_path, 'w') as f_out:
            obj_dict = asdict(self)
            json.dump(obj_dict, f_out, indent=3, cls=NumpyListJsonEncoder)

    @staticmethod
    def load_from_file(file_path, **kwargs):
        with open(file_path) as f_in:
            obj_dict = json.load(f_in)

        if kwargs:
            obj_dict.update(kwargs)

        sim = Simulation(**obj_dict)
        gen_structure.check_genotype_structure(sim.genotype_structure)
        return sim        

    def nn(self):
        # shortcut for creating a list of n None 
        # (with n being the num of agents)
        return [None  for _ in range(self.num_agents)]

    def fill_rand_agent_indexes(self):    
        if self.num_random_pairings in [None, 0]:
            self.random_agent_indexes = None
            return
        
        self.random_agent_indexes = []
        
        # populations have been already shuffled
        # so we take an agent in pop A with corresponding indexes in pop B
        # and rotate on the population B for following simulations
        # 0 -> 0, 0 -> 1, 0 -> 2, .... 1 -> 1, 1 -> 2, 1 -> 3, ...
        # this work also in non dual population because in this case
        # population is split in half        
        if self.dual_population and self.population_index == 1:
            # the way around (dual population)
            # only applies if we want to rerun a simulation focusing on second population
            self.random_agent_indexes = [
                (self.genotype_index - s) % self.population_size
                for s in range(self.num_random_pairings)
            ]
        elif self.genotype_index < self.population_size:
            # stardard case (agent is in the first population)
            # also applies in dual population if population_index == 0
            self.random_agent_indexes = [
                (self.genotype_index + s) % self.population_size
                for s in range(self.num_random_pairings)
            ]
        else:
            # agent is in the second half of the population (not dual population)
            self.random_agent_indexes = [
                (self.genotype_index - self.population_size - s) % self.population_size
                for s in range(self.num_random_pairings)
            ]
        


    def set_agents_genotype_phenotype(self):
        '''
        Split genotype and set phenotype of the two agents
        :param np.ndarray genotypes_pair: sequence with two genotypes (one after the other)
        '''                     
        self.rand_agent_idx = None # index of second agent (if present)        
        if self.num_random_pairings == 0:
            # double genotype
            first_agent_genotype = self.genotype_population[0][self.genotype_index]
            genotypes_pair = first_agent_genotype
            self.genotypes = np.array_split(genotypes_pair, 2)  
        elif self.num_random_pairings is None:
            # isolation
            first_agent_genotype = self.genotype_population[0][self.genotype_index]
            self.genotypes = [first_agent_genotype]
        else:
            # two agents (not double)                                                           
            self.rand_agent_idx = self.random_agent_indexes[self.sim_index]
            if self.dual_population and self.population_index == 1:
                # dual population with current agent in second population 
                self.genotypes = [
                    self.genotype_population[0][self.rand_agent_idx], 
                    self.genotype_population[1][self.genotype_index],                    
                ]            
            elif self.genotype_index < self.population_size:
                # stardard case (agent is in the first population)
                # also applies in dual population with population_index == 0
                self.genotypes = [
                    self.genotype_population[0][self.genotype_index],
                    self.genotype_population[1][self.rand_agent_idx], 
                ]
            else:          
                # agent is in the second half of the population (not dual population)      
                current_agent_genotype_index = self.genotype_index - self.population_size                                    
                self.genotypes = [                        
                    self.genotype_population[0][self.rand_agent_idx], 
                    self.genotype_population[1][current_agent_genotype_index]
                ]                            
        
        self.phenotypes = [{}  for _ in range(self.num_agents)]
        if self.data_record is not None:            
            self.data_record['genotypes'] = self.genotypes
            self.data_record['phenotypes'] = self.phenotypes
        for a,o in enumerate(self.agents):
            o.genotype_to_phenotype(
                self.genotypes[a], 
                phenotype_dict=self.phenotypes[a]
            )
        

    def init_data_record(self):
        if self.data_record is  None:                       
            return                                
        self.data_record['delta_tracker_target'] = [None for _ in range(self.num_trials)]
        self.data_record['target_position'] = [None for _ in range(self.num_trials)]
        self.data_record['target_velocity'] = [None for _ in range(self.num_trials)]
        self.data_record['tracker_position'] = [None for _ in range(self.num_trials)]
        self.data_record['tracker_wheels'] = [None for _ in range(self.num_trials)]
        self.data_record['tracker_velocity'] = [None for _ in range(self.num_trials)]
        self.data_record['tracker_signals'] = [None for _ in range(self.num_trials)]
        self.data_record['agents_motors_control_indexes'] = [None for _ in range(self.num_trials)]
        self.data_record['agents_sensors'] = [self.nn() for _ in range(self.num_trials)]        
        self.data_record['agents_brain_input'] = [self.nn() for _ in range(self.num_trials)]
        self.data_record['agents_brain_state'] = [self.nn() for _ in range(self.num_trials)]
        self.data_record['agents_derivatives'] = [self.nn() for _ in range(self.num_trials)]
        self.data_record['agents_brain_output'] = [self.nn() for _ in range(self.num_trials)]        
        self.data_record['agents_motors'] = [self.nn() for _ in range(self.num_trials)]        
        self.timing.add_time('SIM_init_data', self.tim)

    def init_data_record_trial(self, t):
        if self.data_record is None:            
            return        
        self.data_record['delta_tracker_target'][t] = self.delta_tracker_target       # presaved
        self.data_record['target_position'][t] = self.target_positions           # presaved
        self.data_record['target_velocity'][t] = np.diff(self.target_positions) # presaved
        self.data_record['tracker_position'][t] = np.zeros((self.num_data_points, self.num_dim))
        self.data_record['tracker_wheels'][t] = np.zeros((self.num_data_points, 2))
        self.data_record['tracker_velocity'][t] = np.zeros(self.num_data_points) 
        self.data_record['tracker_signals'][t] = np.zeros((self.num_data_points, 2)) 
        self.data_record['agents_motors_control_indexes'][t] = self.agents_motors_control_indexes
        for a in range(self.num_agents):            
            self.data_record['agents_sensors'][t][a] = np.zeros((self.num_data_points, 2)) # two sensors
            self.data_record['agents_brain_input'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
            self.data_record['agents_brain_state'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
            self.data_record['agents_derivatives'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
            self.data_record['agents_brain_output'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))        
            self.data_record['agents_motors'][t][a] = np.zeros((self.num_data_points, 2)) # two motors  
        self.timing.add_time('SIM_init_trial_data', self.tim)            

    def save_data_record_step(self, t, i):
        if self.data_record is None: 
            return
        self.data_record['tracker_position'][t][i] = self.tracker.position
        self.data_record['tracker_wheels'][t][i] = self.tracker.wheels
        self.data_record['tracker_velocity'][t][i] = self.tracker.velocity
        self.data_record['tracker_signals'][t][i] = self.tracker.signals_strength
        for a,o in enumerate(self.agents):    
            self.data_record['agents_sensors'][t][a][i] = o.sensors
            self.data_record['agents_brain_input'][t][a][i] = o.brain.input                    
            self.data_record['agents_brain_state'][t][a][i] = o.brain.states
            self.data_record['agents_derivatives'][t][a][i] = o.brain.dy_dt
            self.data_record['agents_brain_output'][t][a][i] = o.brain.output
            self.data_record['agents_motors'][t][a][i] = o.motors
            self.timing.add_time('SIM_save_data', self.tim)                            


    def prepare_trial(self, t):

        # init motor controllers
        # agents_motors_control_indexes[0]: which agent's left output is controlling the left motor
        # agents_motors_control_indexes[1]: which agent's right output is controlling the right motor
        self.agents_motors_control_indexes = None
        if self.num_agents == 2:
            if self.isolation_idx is not None:
                # forcing control by one agents
                self.agents_motors_control_indexes = [self.isolation_idx] * 2 # [0,0] or [1,1] 
            elif self.switch_agents_motor_control and t % 2 == 1:
                # invert controller in switch mode on odd trials
                self.agents_motors_control_indexes = [1,0] 
            else:
                self.agents_motors_control_indexes = [0,1] 

        # init deltas
        self.delta_tracker_target = np.zeros(self.num_data_points)        
        # initi all positions and velocities of target
        self.target_positions = self.target.compute_positions(trial = t)
        
        # init data of trial
        self.init_data_record_trial(t) 
                
        # init agents params
        for o in self.agents:
            o.init_params()  

        # init tracker params
        self.tracker.init_params()        
        
        # save trial data at time 0
        self.save_data_record_step(t, 0)        
        
        self.timing.add_time('SIM_prepare_agents_for_trials', self.tim)     

    def compute_brain_input_agents(self):                
        for o in self.agents:
            o.compute_brain_input(self.tracker.signals_strength)
        self.timing.add_time('SIM_compute_brain_input', self.tim)

    def compute_brain_euler_step_agents(self):          
        for o in self.agents:
            o.brain.euler_step()  # this sets agent.brain.output (2-dim vector)
        self.timing.add_time('SIM_euler_step', self.tim)

    def move_tracker_one_step(self):        
        for o in self.agents:
            o.compute_motor_outputs() # compute wheels from motor output        
        if self.num_agents == 1:
            motors = np.copy(self.agents[0].motors)
        else:
            motors = np.array(
                [
                    self.agents[a].motors[i]
                    for i,a in enumerate(self.agents_motors_control_indexes)
                ]
            )
        if self.exclusive_motors_threshold is not None:
            if len(np.where(motors>self.exclusive_motors_threshold)[0]) == 2:
                # when both are more than threshold freeze
                motors = np.zeros(2)
        self.tracker.wheels = motors
        self.tracker.move_one_step()
        self.timing.add_time('SIM_move_one_step', self.tim)  


    #################
    # MAIN FUNCTION
    #################
    def run_simulation(self, genotype_population, genotype_index, random_seed, 
        population_index=0, isolation_idx=None, data_record_list=None):
        '''
        Main function to compute shannon/transfer/sample entropy performace        
        '''
        
        assert population_index==0 or (population_index==1 and self.dual_population), \
            'population_index can be 1 only in dual population mode' 

        self.tim = self.timing.init_tictoc()

        self.genotype_population = genotype_population

        self.genotype_index = genotype_index  
        self.population_index = population_index     
        self.random_state = RandomState(random_seed)        

        if self.split_population():    
            # split pop in two (to allow for pair matching)
            self.genotype_population = np.split(self.genotype_population[0], 2)            
            if self.genotype_index > len(self.genotype_population[0]):
                # keep track current agent is in the second part of population     
                self.population_index = 1  

        self.population_size = len(self.genotype_population[0])
        self.isolation_idx = isolation_idx
        if self.isolation_idx is not None:
            assert self.num_agents==2, \
                'can only force isolation if simulatin is run with 2 agents'
        
        self.fill_rand_agent_indexes() # random_agent_indexes

        num_simulations = 1 if self.random_agent_indexes is None \
            else max(1, len(self.random_agent_indexes))
        
        sim_performances = []

        # SIMULATIONS START

        self.agents_similarity = [None for _ in range(num_simulations)]

        for self.sim_index in range(num_simulations):

            self.data_record = None 
            if data_record_list is not None: 
                self.data_record = {}
                data_record_list.append(self.data_record)

            self.set_agents_genotype_phenotype()    
            self.timing.add_time('SIM_init_agent_phenotypes', self.tim)    

            trial_performances = []        

            # INITIALIZE DATA RECORD
            self.init_data_record()        

            # TRIALS START
            for t in range(self.num_trials):

                # setup trial
                self.prepare_trial(t)            
                            
                for i in range(1, self.num_data_points):                

                    # 1) Agent senses strength of emitter from the two sensors
                    self.tracker.compute_signal_strength_and_delta_target(self.target_positions[i])
                    self.delta_tracker_target[i] = self.tracker.delta_target

                    # 2) compute brain input
                    self.compute_brain_input_agents()

                    # 3) Update agent's neural system
                    self.compute_brain_euler_step_agents()

                    # 4) Move one step  agents
                    self.move_tracker_one_step()
                    
                    self.save_data_record_step(t, i)             

                # performance_t = - np.mean(np.abs(self.delta_tracker_target)) / self.target_env_width
                performance_t = MAX_MEAN_DISTANCE - np.mean(np.abs(self.delta_tracker_target))

                trial_performances.append(performance_t)

            # TRIALS END

            # returning mean performances between all trials
            exp_perf = np.mean(trial_performances)

            self.agents_similarity[self.sim_index] = self.get_genotypes_similarity()

            if self.data_record:
                self.data_record['info'] = {
                    'rand_agent_index': self.rand_agent_idx,
                    'genotype_similarity': self.agents_similarity[self.sim_index],
                    'trials_performances': trial_performances,
                    'experiment_performance': exp_perf,                    
                }

            sim_performances.append(exp_perf)

        # SIMULATIONS END
        
        total_performance = np.mean(sim_performances)
        return total_performance, sim_performances, self.random_agent_indexes


    ##################
    # EVAL FUNCTION
    ##################

    def evaluate(self, populations, random_seed):    

        population_size = len(populations[0])        
        
        if self.split_population():     
            assert population_size % 2 == 0
            # we only run the first half (because of matched pairs)
            population_size = int(population_size/2)

        if self.num_cores > 1:
            # run parallel job            
            sim_array = [Simulation(**asdict(self)) for _ in range(self.num_cores)]
            run_result = Parallel(n_jobs=self.num_cores)( 
                delayed(sim_array[i%self.num_cores].run_simulation)(populations, i, random_seed) \
                for i in range(population_size)
            )
        else:
            # single core
            run_result = [
                self.run_simulation(populations, i, random_seed)
                for i in range(population_size)
            ]
        
        if self.num_random_pairings!=None and self.num_random_pairings>0:
            # compute performance of the second (half of the) population
            first_half_performances = np.zeros((population_size))
            second_half_performances = np.zeros((self.num_random_pairings, population_size)) # stor individual performnaces with each pairing and then compute mean
            for i, r in enumerate(run_result):
                perf_tot, perf_sim_list, rand_idx_list = r
                first_half_performances[i] = perf_tot # average of perf of the i-th agent of first population paired with n other agents of the second population
                for j, (perf_sim, rand_idx) in enumerate(zip(perf_sim_list, rand_idx_list)):
                    second_half_performances[j][rand_idx] = perf_sim # adding single sim performance to second agent
            second_half_performances = np.mean(second_half_performances, axis=0) # average performances on second population
            performances = np.array([first_half_performances, second_half_performances])
            if not self.dual_population:
                # joined thw two half performances in one
                performances = np.concatenate(performances)
        else:
            performances = [p[0] for p in run_result]
        

        # todo: return array of performances based on number of populations
        return performances


    def normalize_performance(self, performance):
        '''
        Returns the normalized performance
        such that best performance == 0
        '''
        return MAX_MEAN_DISTANCE - performance


# --- END OF SIMULATION CLASS


# TEST

def get_simulation_data_from_agent(gen_struct, genotype, rs, num_dim=1):
    sim = Simulation(
        genotype_structure=gen_struct,
        num_dim=num_dim
    )
    data_record_list = []
    run_result = sim.run_simulation(        
        genotype_population=[[genotype]], 
        genotype_index=0,
        random_seed=utils.random_int(rs),
        population_index=0,
        data_record_list=data_record_list,        
    )    
    return run_result, sim, data_record_list

def get_simulation_data_from_random_agent(gen_struct, rs, num_dim=1):    
    from pyevolver.evolution import Evolution    
    gen_size = gen_structure.get_genotype_size(gen_struct)
    random_genotype = Evolution.get_random_genotype(rs, gen_size)    
    return get_simulation_data_from_agent(gen_struct, random_genotype, rs, num_dim)

def get_simulation_data_from_filled_agent(gen_struct, value, rs, num_dim=1):    
    gen_size = gen_structure.get_genotype_size(gen_struct)
    genotype = np.full(gen_size, value)
    return get_simulation_data_from_agent(gen_struct, genotype, rs, num_dim)
    

def test_simulation():
    default_gen_structure = gen_structure.DEFAULT_GEN_STRUCTURE(2)
    rs = RandomState(3)
    run_result, _, data_record_list = get_simulation_data_from_random_agent(default_gen_structure, rs)    
    perf = run_result[0]
    print("Performance: ", perf)
    utils.save_json_numpy_data(data_record_list, 'data/simulation.json')    

def ger_worst_performance(num_iter):
    worst = MAX_MEAN_DISTANCE
    default_gen_structure = gen_structure.DEFAULT_GEN_STRUCTURE(2)    
    rs = RandomState(None)
    for _ in range(num_iter):
        run_result, _, _ = get_simulation_data_from_random_agent(default_gen_structure, rs)
        perf = run_result[0]
        if perf < worst:
            worst = perf
            print('Worst perf: ', worst)
    

if __name__ == "__main__":
    test_simulation()
    # ger_worst_performance(100)