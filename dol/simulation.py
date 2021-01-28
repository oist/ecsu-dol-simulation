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
from dol import gen_structure
from dol import utils
# from dol.utils import assert_string_in_values

@dataclass
class Simulation:        
    genotype_structure: Dict = field(default_factory=lambda:gen_structure.DEFAULT_GEN_STRUCTURE(2))
    num_agents: int = 1 # 1 or 2 (2 agents controlling the wheels)
    num_brain_neurons: int = None  # initialized in __post_init__
    brain_step_size: float = 0.1
    num_trials: int = 6
    trial_duration: int = 80    
    num_cores: int = 1
    timeit: bool = False

    target_env_width = 400
    target_trial_vel: list = field(default_factory=lambda:[2, -1, -2, 1, -3, 3])
    target_trial_start_pos: list = field(default_factory=lambda:[-10, 0, 20, -20, 10, 0])
    target_trial_delta_bnd: list = field(default_factory=lambda:[0, 20, 10, 20, 0, 10])
    target_random_pos_max_value: float = None # upper bnd of start self.pos and delta bnd
    target_random_vel_max_value: float = None # upper bnd of start self.pos and delta bnd

    def __post_init__(self):          

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

        self.tracker = Tracker()

        self.init_target()

        self.timing = Timing(self.timeit)        

        self.__check_params__()

    def __check_params__(self):
        pass

    def init_target(self):
        self.target = Target(
            num_data_points = self.num_data_points,
            env_width = self.target_env_width,
            trial_vel = self.target_trial_vel,
            trial_start_pos = self.target_trial_start_pos,
            trial_delta_bnd = self.target_trial_delta_bnd,
            random_pos_max_value = self.target_random_pos_max_value,
            random_vel_max_value = self.target_random_vel_max_value
        )

    def init_random_target(self, pos_max_value=20, vel_max_value=3):
        self.target_trial_vel = None
        self.target_trial_start_pos = None
        self.target_trial_delta_bnd = None
        self.target_random_pos_max_value = pos_max_value
        self.target_random_vel_max_value = vel_max_value
        self.init_target()

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

    def set_agents_genotype_phenotype(self):
        '''
        Split genotype and set phenotype of the two agents
        :param np.ndarray genotypes_pair: sequence with two genotypes (one after the other)
        '''                
        genotypes = [self.genotype_population[self.genotype_index]]
        if self.num_agents == 2:
            genotypes.append(
                self.genotype_population[self.rand_agent_indexes[self.sim_index]], 
            )        
        if self.data_record is not None:
            phenotypes = [{}  for _ in range(self.num_agents)]
            self.data_record['genotypes'] = genotypes
            self.data_record['phenotypes'] = phenotypes
        else:
            phenotypes = self.nn()
        for a,o in enumerate(self.agents):
            o.genotype_to_phenotype(
                genotypes[a], 
                phenotype_dict=phenotypes[a]
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
        self.data_record['agents_brain_input'] = [self.nn() for _ in range(self.num_trials)]
        self.data_record['agents_brain_state'] = [self.nn() for _ in range(self.num_trials)]
        self.data_record['agents_derivatives'] = [self.nn() for _ in range(self.num_trials)]
        self.data_record['agents_brain_output'] = [self.nn() for _ in range(self.num_trials)]        
        self.timing.add_time('SIM_init_data', self.tim)

    def init_data_record_trial(self, t):
        if self.data_record is None:            
            return
        self.data_record['delta_tracker_target'][t] = self.delta_tracker_target  # presaved
        self.data_record['target_position'][t] = self.target_positions           # presaved
        self.data_record['target_velocity'][t] = np.diff(self.target_positions) # presaved
        self.data_record['tracker_position'][t] = np.zeros(self.num_data_points)
        self.data_record['tracker_wheels'][t] = np.zeros((self.num_data_points, 2))
        self.data_record['tracker_velocity'][t] = np.zeros(self.num_data_points) 
        self.data_record['tracker_signals'][t] = np.zeros((self.num_data_points, 2)) 
        for a in range(self.num_agents):            
            self.data_record['agents_brain_input'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
            self.data_record['agents_brain_state'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
            self.data_record['agents_derivatives'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
            self.data_record['agents_brain_output'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))        
        self.timing.add_time('SIM_init_trial_data', self.tim)            

    def save_data_record_step(self, t, i):
        if self.data_record is None: 
            return
        self.data_record['tracker_position'][t][i] = self.tracker.position
        self.data_record['tracker_wheels'][t][i] = self.tracker.wheels
        self.data_record['tracker_velocity'][t][i] = self.tracker.velocity
        self.data_record['tracker_signals'][t][i] = self.tracker_signals_strength
        for a,o in enumerate(self.agents):            
            self.data_record['agents_brain_input'][t][a][i] = o.brain.input                    
            self.data_record['agents_brain_state'][t][a][i] = o.brain.states
            self.data_record['agents_derivatives'][t][a][i] = o.brain.dy_dt
            self.data_record['agents_brain_output'][t][a][i] = o.brain.output
            self.timing.add_time('SIM_save_data', self.tim)                            


    def prepare_trial(self, t):

        # init deltas
        self.delta_tracker_target = np.zeros(self.num_data_points)        

        # init signal strengths (vision)
        self.tracker_signals_strength = np.zeros(2) # init signal strength 

        # initi all positions and velocities of target
        self.target_positions = self.target.compute_positions(            
            trial = t,
            rs = self.random_state
        )
        
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

    def compute_tracker_signals_strength(self, i):
        delta = self.target_positions[i] - self.tracker.position        
        delta_abs = np.abs(delta)
        if delta_abs < 1:
            # consider tracker and target overlapping -> max signla left and right sensor
            self.tracker_signals_strength = np.ones(2)
        else:
            signal_index = 1 if delta > 0 else 0 # right or left
            self.tracker_signals_strength = np.zeros(2)
            self.tracker_signals_strength[signal_index] = 1/delta_abs
        # store delta
        self.delta_tracker_target[i] = delta

    def compute_brain_input_agents(self):                
        if self.num_agents == 1:
            self.agents[0].compute_brain_input(self.tracker_signals_strength)
        else:
            for a,o in enumerate(self.agents):
                signal_strength = np.zeros(2)
                signal_strength[a] = self.tracker_signals_strength[a]
                o.compute_brain_input(signal_strength)
        self.timing.add_time('SIM_compute_brain_input', self.tim)

    def compute_brain_euler_step_agents(self):          
        for o in self.agents:
            o.brain.euler_step()  # this sets agent.brain.output (2-dim vector)
        self.timing.add_time('SIM_euler_step', self.tim)

    def move_tracker_one_step(self):        
        for o in self.agents:
            o.compute_motor_outputs() # compute wheels from motor output        
        if self.num_agents == 1:
            self.tracker.wheels = np.copy(self.agents[0].motors)
        else:
            self.tracker.wheels = np.array([self.agents[0].motors[0],self.agents[1].motors[1]])        
        self.tracker.move_one_step()
        self.timing.add_time('SIM_move_one_step', self.tim)  


    #################
    # MAIN FUNCTION
    #################
    def run_simulation(self, genotype_population=None, genotype_index=None,
        rnd_seed=0, data_record=None):
        '''
        Main function to compute shannon/transfer/sample entropy performace        
        '''

        self.tim = self.timing.init_tictoc()

        self.genotype_population = genotype_population
        self.genotype_index = genotype_index
        self.random_state = RandomState(rnd_seed)
        self.rand_agent_indexes = []

        # fill rand_agent_indexes with n indexes i
        if self.num_agents == 2:
            assert False
            # while len(self.rand_agent_indexes) != self.num_random_pairings:
            #     next_rand_index = self.random_state.randint(len(self.genotype_population))
            #     if next_rand_index != self.genotype_index:
            #         self.rand_agent_indexes.append(next_rand_index)

        self.data_record = data_record 

        if self.genotype_population is not None:            
            self.set_agents_genotype_phenotype()    
            self.timing.add_time('SIM_init_agent_phenotypes', self.tim)    

        trial_performances = []        

        # INITIALIZE DATA RECORD
        self.init_data_record()        

        # EXPERIMENT START
        for t in range(self.num_trials):

            # setup trial
            self.prepare_trial(t)            
                        
            for i in range(1, self.num_data_points):                

                # 1) Agent senses strength of emitter from the two sensors
                self.compute_tracker_signals_strength(i) # computes self.tracker_signals_strength

                # 2) compute brain input
                self.compute_brain_input_agents()

                # 3) Update agent's neural system
                self.compute_brain_euler_step_agents()

                # 4) Move one step  agents
                self.move_tracker_one_step()
                
                self.save_data_record_step(t, i)             

            performance_t = - np.mean(np.abs(self.delta_tracker_target))

            trial_performances.append(performance_t)

        # EXPERIMENT END

        # returning mean performances between all trials
        exp_perf = np.mean(trial_performances)
        if self.data_record:
            self.data_record['summary'] = {
                'rand_agent_indexes': self.rand_agent_indexes,
                'trials_performances': trial_performances,
                'experiment_performance': exp_perf
            }
        
        return exp_perf

    '''
    POPULATION EVALUATION FUNCTION
    '''
    def evaluate(self, population, random_seeds):                
        population_size = len(population)
        assert population_size == len(random_seeds)

        if self.num_cores > 1:
            # run parallel job            
            sim_array = [Simulation(**asdict(self)) for _ in range(self.num_cores)]
            performances = Parallel(n_jobs=self.num_cores)( 
                delayed(sim_array[i%self.num_cores].run_simulation)(population, i, rnd_seed) \
                for i, (_, rnd_seed) in enumerate(zip(population, random_seeds))
            )
        else:
            # single core
            performances = [
                self.run_simulation(population, i, rnd_seed)
                for i, (_, rnd_seed) in enumerate(zip(population, random_seeds))
             ]

        return performances

# TEST

def get_simulation_data_from_random_agent():
    from pyevolver.evolution import Evolution
    default_gen_structure = gen_structure.DEFAULT_GEN_STRUCTURE(2)
    gen_size = gen_structure.get_genotype_size(default_gen_structure)
    random_genotype = Evolution.get_random_genotype(RandomState(None), gen_size)    
    
    sim = Simulation()
    data_record = {}
    perf = sim.run_simulation(
        genotype_population=[random_genotype], 
        genotype_index=0,
        rnd_seed=0, 
        data_record=data_record
    )
    print("Performance: ", perf)
    return sim, data_record

def test_simulation():
    _, data_record = get_simulation_data_from_random_agent()
    utils.save_json_numpy_data(data_record, 'data/simulation.json')    

if __name__ == "__main__":
    test_simulation()