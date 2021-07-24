"""
Main code for experiment simulation.
"""

from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Tuple, List
import json
import numpy as np
from numpy.random import RandomState
from joblib import Parallel, delayed
from pyevolver.json_numpy import NumpyListJsonEncoder
from pyevolver.timing import Timing
from dol import tracker2d
from dol.agent import Agent
from dol.tracker import Tracker
from dol.target import Target
from dol.target2d import Target2D
from dol.tracker2d import Tracker2D
from dol import gen_structure
from dol import utils

@dataclass
class Simulation:
    # the genotype structure  
    genotype_structure: Dict = field(default_factory = \
        lambda:gen_structure.DEFAULT_GEN_STRUCTURE(1,2))  

    num_pop: int = 1 # number of populations

    num_dim: int = 1  # number of dimensions (1 or 2)

    # random pairings
    num_random_pairings: int = None
    # None -> agents are alone in the simulation (default)
    # 0    -> agents are evolved in pairs: a genotype contains a pair of agents
    # N>0  -> each agent will go though a simulation with N other agents

    # for back compatibility - to be removed (now using motor_control_mode)
    # False -> SEPARATE
    # Ture -> SWITCH
    switch_agents_motor_control: bool = False

    # for back compatibility - to be removed (now using num_pop)
    dual_population: float = False 

    motor_control_mode: str = None # 'SEPARATE' # SEPARATE, SWITCH, OVERLAP
    # when num_agents is 2 this decides whether
    # None: not applicable (if self.num_agents==1)
    # SEPARATE: across trials the first agent always control the left motor and the second the right
    # SWITCH: the two agents switch control of L/R motors in different trials
    # OVERLAP: both agents control L/R motors (for a factor o half)

    exclusive_motors_threshold: float = None

    max_mean_distance = 10000 # for turning min to max in fitness function and normalization

    brain_step_size: float = 0.1
    num_trials: int = 4
    trial_duration: int = 50
    num_cores: int = 1
    random_seed: int = 0
    timeit: bool = False

    def __post_init__(self):

        self.num_agents = 1 if self.num_random_pairings is None else 2

        if self.dual_population:            
            # for back compatibility
            self.num_pop = 2 
            self.motor_control_mode = 'SWITCH' if self.switch_agents_motor_control else 'SEPARATE'
        elif self.switch_agents_motor_control:            
            # for back compatibility
            self.motor_control_mode = None if self.num_agents==1 else 'SEPARATE'

        self.__check_params__()        

        self.num_sensors_motors = 2 * self.num_dim

        self.num_brain_neurons = gen_structure.get_num_brain_neurons(self.genotype_structure)
        self.num_data_points = int(self.trial_duration / self.brain_step_size)

        self.agents = [
            Agent(
                self.num_dim,
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
        assert self.num_random_pairings is None or self.num_random_pairings >= 0, \
            "num_random_pairings must be None or >= 0"

        assert self.num_pop==1 or self.num_random_pairings > 0, \
            "In multiple populations, num_random_pairings must be > 0"

        assert self.num_agents==2 or self.motor_control_mode==None, \
            "With one agent motor_control_mode must be None"

        assert self.num_agents==1 or self.motor_control_mode!=None, \
            "With two agents motor_control_mode must not be None"
        
        utils.assert_string_in_values(
            self.motor_control_mode, 
            'motor_control_mode',
            [None, 'SEPARATE', 'SWITCH', 'OVERLAP']
        )        

        assert self.num_pop<=2 or self.num_pop==self.num_random_pairings+1, \
            "In multiple populations, num_pop must be equal to 2 \
            (dual population) or num_random_pairings+1 \
            (each agent is pair with each agent at the same index of other population)"

        assert self.num_dim == 1 or self.exclusive_motors_threshold is None, \
            "In 2d mode exclusive_motors mode is not appropriate"
            # TODO: double check this

    def split_population(self):
        # when population will be split in two for computing random pairs matching
        return self.num_pop==1 and \
               self.num_random_pairings is not None and \
               self.num_random_pairings > 0

    def init_tracker(self):
        if self.num_dim == 1:
            self.tracker = Tracker()
        else:
            assert self.num_dim == 2
            self.tracker = Tracker2D()

    def init_target(self, rs=None):
        if self.num_dim == 1:
            self.target = Target(
                self.num_data_points,
                self.num_trials,
                rs
            )
        else:
            assert self.num_dim == 2
            self.target = Target2D(
                self.num_data_points,
                self.num_trials,
                rs
            )

    def get_genotypes_distance(self):
        if self.num_agents == 1:
            return None
        return utils.genotype_distance(
            self.genotypes[0], 
            self.genotypes[1]
        )

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
        if 'motor_control_mode' not in obj_dict:
            # for back compatibility - old simulations used different threshold
            sim.max_mean_distance = 5000            

        gen_structure.check_genotype_structure(sim.genotype_structure)
        return sim

    def fill_paired_agents_indexes(self, exaustive_pairs):
        if self.num_random_pairings in [None, 0]:
            self.paired_agents_sims_pop_idx = None
            return

        self.paired_agents_sims_pop_idx = []

        if self.num_pop > 2:
            if exaustive_pairs:
                self.paired_agents_sims_pop_idx = [
                    (pop, self.genotype_index)
                    for pop in range(0, self.num_random_pairings+1)
                    if pop != self.population_index # exclude itself
                ]
            else:
                self.paired_agents_sims_pop_idx = [
                    (pop, self.genotype_index)
                    for pop in range(self.population_index+1, self.num_random_pairings+1)
                ]
        # populations have been already shuffled
        # so we take an agent in pop A with corresponding indexes in pop B
        # and rotate on the population B for following simulations
        # 0 -> 0, 0 -> 1, 0 -> 2, .... 1 -> 1, 1 -> 2, 1 -> 3, ...
        # this works also in non dual population because in this case
        # population is split in half        
        elif self.population_index == 1:
            # agent is in the second half of the population
            # only applies if we want to rerun a simulation focusing on agent in second population
            self.paired_agents_sims_pop_idx = [
                (0, (self.genotype_index - s) % self.population_size)
                for s in range(self.num_random_pairings)
            ]
        else:
            # stardard case (agent is in the first population)
            # also applies in dual population if population_index == 0
            self.paired_agents_sims_pop_idx = [
                (1, (self.genotype_index + s) % self.population_size)
                for s in range(self.num_random_pairings)
            ]

    def set_agents_genotype_phenotype(self):
        '''
        Split genotype and set phenotype of the two agents
        It sets self.paired_agent_pop_idx (tuple with pop and index of paired agent, if it exists)     
        '''
        if self.num_random_pairings is None:
            # single genotype
            first_agent_genotype = self.genotype_population[0][self.genotype_index]
            self.genotypes = [first_agent_genotype]
            self.paired_agent_pop_idx = None # no paired agent
        elif self.num_random_pairings == 0:
            # double genotype
            first_agent_genotype = self.genotype_population[0][self.genotype_index]
            genotypes_pair = first_agent_genotype
            self.genotypes = np.array_split(genotypes_pair, 2)        
            self.paired_agent_pop_idx = (0, self.genotype_index)
        else:
            # two agents
            self.paired_agent_pop_idx = self.paired_agents_sims_pop_idx[self.sim_index]            
            p_pop, p_idx = self.paired_agent_pop_idx
            if self.population_index < p_pop:
                self.genotypes = [
                    self.genotype_population[self.population_index][self.genotype_index],
                    self.genotype_population[p_pop][p_idx],
                ]
            else:
                self.genotypes = [
                    self.genotype_population[p_pop][p_idx],
                    self.genotype_population[self.population_index][self.genotype_index]                    
                ]

        self.phenotypes = [{} for _ in range(self.num_agents)]
        if self.data_record is not None:
            self.data_record['genotypes'] = self.genotypes
            self.data_record['phenotypes'] = self.phenotypes
        for a, o in enumerate(self.agents):
            o.genotype_to_phenotype(
                self.genotypes[a],
                phenotype_dict=self.phenotypes[a]
            )

    def init_data_record(self):
        num_wheels = 2 if (self.num_dim ==1 or not tracker2d.XY_MODE) else 4
        if self.data_record is None:
            return
        self.data_record['delta_tracker_target'] = np.zeros((self.num_trials, self.num_data_points))
        self.data_record['target_position'] = \
            np.zeros((self.num_trials, self.num_data_points)) if self.num_dim==1 \
            else np.zeros((self.num_trials, self.num_data_points, self.num_dim))
        self.data_record['tracker_position'] = \
            np.zeros((self.num_trials, self.num_data_points)) if self.num_dim==1 \
            else np.zeros((self.num_trials, self.num_data_points, self.num_dim))
        self.data_record['tracker_angle'] = np.zeros((self.num_trials, self.num_data_points))
        self.data_record['tracker_wheels'] = np.zeros((self.num_trials, self.num_data_points, num_wheels))
        self.data_record['tracker_velocity'] = np.zeros((self.num_trials, self.num_data_points))
        self.data_record['tracker_signals'] = np.zeros((self.num_trials, self.num_data_points, self.num_sensors_motors))
        self.data_record['agents_motors_control_indexes'] = [None for _ in range(self.num_trials)]
        self.data_record['agents_sensors'] = np.zeros((self.num_trials, self.num_agents, self.num_data_points, self.num_sensors_motors))
        self.data_record['agents_brain_input'] = np.zeros((self.num_trials, self.num_agents, self.num_data_points, self.num_brain_neurons))
        self.data_record['agents_brain_state'] = np.zeros((self.num_trials, self.num_agents, self.num_data_points, self.num_brain_neurons))
        self.data_record['agents_derivatives'] = np.zeros((self.num_trials, self.num_agents, self.num_data_points, self.num_brain_neurons))
        self.data_record['agents_brain_output'] = np.zeros((self.num_trials, self.num_agents, self.num_data_points, self.num_brain_neurons))
        self.data_record['agents_motors'] = np.zeros((self.num_trials, self.num_agents, self.num_data_points, self.num_sensors_motors))
        self.timing.add_time('SIM_init_data', self.tim)

    def save_data_record_trial(self, t):
        if self.data_record is None:
            return
        self.data_record['delta_tracker_target'][t] = self.delta_tracker_target  # presaved
        self.data_record['target_position'][t] = self.target_positions  # presaved
        self.data_record['agents_motors_control_indexes'][t] = self.agents_motors_control_indexes

        if self.with_ghost:
            for k in ['agents_sensors', 'agents_brain_input', 'agents_brain_state',
                      'agents_derivatives', 'agents_brain_output', 'agents_motors']:
                self.data_record[k][t][self.ghost_index] = self.original_data_record[k][t][self.ghost_index]

        self.timing.add_time('SIM_init_trial_data', self.tim)

    def save_data_record_step(self, t, i):
        if self.data_record is None:
            return

        self.data_record['tracker_position'][t][i] = self.tracker.position
        self.data_record['tracker_angle'][t][i] = self.tracker.angle
        self.data_record['tracker_wheels'][t][i] = self.tracker.wheels
        self.data_record['tracker_velocity'][t][i] = self.tracker.velocity
        self.data_record['tracker_signals'][t][i] = self.tracker.signals_strength
        
        for a, o in enumerate(self.agents):
            if a == self.ghost_index:
                continue
            self.data_record['agents_sensors'][t][a][i] = o.sensors
            self.data_record['agents_brain_input'][t][a][i] = o.brain.input
            self.data_record['agents_brain_state'][t][a][i] = o.brain.states
            self.data_record['agents_derivatives'][t][a][i] = o.brain.dy_dt
            self.data_record['agents_brain_output'][t][a][i] = o.brain.output
            self.data_record['agents_motors'][t][a][i] = o.motors
        self.timing.add_time('SIM_save_data', self.tim)

    def prepare_trial(self, t):

        # init motor controllers
        # motors/wheels indexes in sequence: left, right (, down, up) 
        # agents_motors_control_indexes[0]: which agent is controlling the left wheel
        # agents_motors_control_indexes[1]: which agent is controlling the right motor
        # in addition, if num_dim is 2
        # agents_motors_control_indexes[2]: which agent is controlling the down wheel
        # agents_motors_control_indexes[3]: which agent is controlling the up wheel
        
        if self.num_agents == 1:
            # single agent
            self.agents_motors_control_indexes = None # only relevant for 2 agents
        else:
            # 2 agents            
            if self.motor_control_mode=='SWITCH':
                if t % 2 == 0: # (0,2) - (first, third)
                    if self.num_dim == 1:
                        self.agents_motors_control_indexes = [0, 1]
                    else:
                        # 2d - first agent controls horizontal wheels, second agents vertical wheels
                        self.agents_motors_control_indexes = [0, 0, 1, 1]
                else:
                    # invert controller in switch mode on odd trial indexes (1,3) - (second, forth)
                    if self.num_dim == 1:
                        self.agents_motors_control_indexes = [1, 0]
                    else:
                        # 2d - first agent controls vertical wheels, second agents horizontal wheels
                        self.agents_motors_control_indexes = [1, 1, 0, 0]
            elif self.motor_control_mode=='SEPARATE':
                if self.num_dim == 1:
                    # across all trials first agent controls left motor, second agent controls right motor
                    self.agents_motors_control_indexes = [0, 1]
                else:
                    # 2d - # across all trials first agent controls horizontal wheels, second agents vertical wheels
                    self.agents_motors_control_indexes = [0, 0, 1, 1]
            else:
                # self.motor_control_mode=='OVERLAP'
                # both agents control both motors (for a factor of half)
                self.agents_motors_control_indexes = None 
            
        
        # init deltas
        self.delta_tracker_target = np.zeros(self.num_data_points)
        # initi all positions and velocities of target
        self.target_positions = self.target.compute_positions(trial=t)        

        # init agents params
        for i, o in enumerate(self.agents):
            if i==self.ghost_index: 
                continue
            o.init_params(self.init_ctrnn_state)

            # init tracker params
        self.tracker.init_params_trial(t)
        # self.tracker.position = np.copy(self.target_positions[0])

        # save trial data at time 0
        # self.save_data_record_step(t, 0)

        self.timing.add_time('SIM_prepare_agents_for_trials', self.tim)

    def compute_brain_input_agents(self):
        for i, o in enumerate(self.agents):
            if i==self.ghost_index: 
                continue
            o.compute_brain_input(self.tracker.signals_strength)
        self.timing.add_time('SIM_compute_brain_input', self.tim)

    def compute_brain_euler_step_agents(self):
        for i, o in enumerate(self.agents):
            if i==self.ghost_index: 
                continue
            o.brain.euler_step()  # this sets agent.brain.output (2-dim vector)
        self.timing.add_time('SIM_euler_step', self.tim)

    def compute_motor_outputs_and_wheels(self):        
        
        for i, o in enumerate(self.agents):
            if i==self.ghost_index: 
                o.motors = self.played_back_ghost_motors
                continue
            o.compute_motor_outputs()  # compute wheels from motor output
        
        if self.num_agents == 1:
            motors = np.copy(self.agents[0].motors)
        else:
            # 2 agents
            if self.motor_control_mode == 'OVERLAP':
                motors = np.array([ # [[a1_m1, a1_m2],[a2_m1, a2_m2]
                    self.agents[a].motors
                    for a in range(2)
                ]).mean(axis=0) # mean across rows
            else:                
                motors = np.array(
                    [
                        self.agents[a].motors[i]
                        for i, a in enumerate(self.agents_motors_control_indexes)
                    ]
                )
        if self.exclusive_motors_threshold is not None:
            if len(np.where(motors > self.exclusive_motors_threshold)[0]) == 2:
                # when both are more than threshold freeze
                motors = np.zeros(2)
        
        self.tracker.wheels = motors
        
        self.timing.add_time('SIM_move_one_step', self.tim)

    #################
    # MAIN FUNCTION
    #################
    def run_simulation(self, genotype_population, genotype_index, random_seed,
                       population_index=0, exaustive_pairs=False, 
                       init_ctrnn_state=0., data_record_list=None,
                       ghost_index=None, original_data_record_list=None):
        '''
        Main function to compute shannon/transfer/sample entropy performace        
        param exaustive_pairs (bool): if to compute all pairs 
              by default False, during evolution for optimization purporses
              (all pairs are recombined afterwards in evalutate() )
        param ghost_index: run simulation with ghost agent at the given index
              using its played back data specified in original_data_record_list
        '''

        assert population_index == 0 or self.num_pop > 1, \
            'population_index can be != 1 only if num_pop > 1'

        self.tim = self.timing.init_tictoc()

        self.genotype_population = genotype_population

        self.genotype_index = genotype_index
        self.population_index = population_index
        self.random_state = RandomState(random_seed)

        if self.split_population():
            # split pop in two (to allow for pair matching)
            self.genotype_population = np.split(self.genotype_population[0], 2)
            if self.genotype_index >= len(self.genotype_population[0]):
                # keep track current agent is in the second part of population     
                self.genotype_index = self.genotype_index - len(self.genotype_population[0])
                self.population_index = 1

        self.population_size = len(self.genotype_population[0])
        self.init_ctrnn_state = init_ctrnn_state

        self.ghost_index = ghost_index
        self.with_ghost = ghost_index is not None

        if self.with_ghost:
            assert original_data_record_list is not None, \
                'If you want to use ghost simulation you have to provide me with the original_data_record_list'
            assert self.num_agents==2, \
                'Cannot have ghost in isolated case (single agent)'

        self.fill_paired_agents_indexes(exaustive_pairs)  # paired_agents_sims_pop_idx

        num_simulations = 1 if self.paired_agents_sims_pop_idx is None \
            else max(1, len(self.paired_agents_sims_pop_idx))

        sim_performances = []

        # SIMULATIONS START

        self.agents_genotype_distance = [None for _ in range(num_simulations)]

        for self.sim_index in range(num_simulations):

            if self.with_ghost:
                self.original_data_record = original_data_record_list[self.sim_index]
            
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

                if self.with_ghost:
                    trial_ghost_motors = self.original_data_record['agents_motors'][t][self.ghost_index]

                # replace with range(1, self.num_data_points) to reproduce old results
                for i in range(self.num_data_points): 
                    if self.with_ghost:
                        self.played_back_ghost_motors = trial_ghost_motors[i]

                    # 1) Agent senses strength of emitter from the two sensors
                    self.tracker.compute_signal_strength_and_delta_target(self.target_positions[i])
                    
                    self.delta_tracker_target[i] = self.tracker.delta_target

                    # 2) compute brain input
                    self.compute_brain_input_agents()

                    # 3) Update agent's neural system
                    self.compute_brain_euler_step_agents()

                    # 4) Compute agent's motor output and wheels
                    self.compute_motor_outputs_and_wheels()   

                    self.save_data_record_step(t, i)

                    # 5) Move tracker one step
                    self.tracker.move_one_step()                 

                # performance_t = - np.mean(np.abs(self.delta_tracker_target)) / self.target_env_width
                performance_t = self.max_mean_distance - np.mean(np.abs(self.delta_tracker_target))
                assert performance_t >= 0, f"Found performance trial < 0: {performance_t}"

                trial_performances.append(performance_t)

                # save data of trial
                self.save_data_record_trial(t)

            # TRIALS END

            # returning mean performances between all trials
            exp_perf = np.mean(trial_performances)

            self.agents_genotype_distance[self.sim_index] = self.get_genotypes_distance()

            if self.data_record:
                self.data_record.update({
                    'current_agent_pop_idx': (self.population_index, self.genotype_index),
                    'paired_agent_pop_idx': self.paired_agent_pop_idx,                    
                    'genotype_distance': self.agents_genotype_distance[self.sim_index],
                    'trials_performances': trial_performances,
                    'sim_performance': exp_perf,
                })

            sim_performances.append(exp_perf)

        # SIMULATIONS END

        total_performance = np.mean(sim_performances)
        return total_performance, sim_performances, self.paired_agents_sims_pop_idx

    '''
    Run Simulation of agents of a give index across populations
    Only applies if num_pop > 2
    '''
    def run_pop_simulations(self, genotype_population, genotype_index, random_seed):
        pop_results = []
        for pop_idx in range(self.num_pop-1): # exlude last
            pop_results.append(
                self.run_simulation(
                    genotype_population=genotype_population, 
                    genotype_index=genotype_index, 
                    random_seed=random_seed, 
                    population_index=pop_idx
                )
            )
        return pop_results        


    ##################
    # EVAL FUNCTION
    ##################
    def evaluate(self, populations, random_seed):

        population_size = len(populations[0])

        sim_copy = Simulation(**asdict(self))

        if self.num_pop <= 2:
            if self.split_population():
                assert population_size % 2 == 0
                # we only run the first half (because of matched pairs)
                population_size = int(population_size / 2)

            if self.num_cores > 1:
                # run parallel job            
                run_result = Parallel(n_jobs=self.num_cores)(
                    delayed(sim_copy.run_simulation)(populations, i, random_seed) \
                    for i in range(population_size)
                )
            else:
                # single core
                run_result = [
                    sim_copy.run_simulation(populations, i, random_seed)
                    for i in range(population_size)
                ]

            if self.num_random_pairings != None and self.num_random_pairings > 0:
                # compute performance of the second (half of the) population
                # dual or split population
                first_half_performances = np.zeros((population_size))
                
                # store individual performnaces with each pairing and then compute mean
                second_half_performances = \
                    np.zeros(
                        (self.num_random_pairings, population_size)
                    )  
                    
                for i, r in enumerate(run_result):
                    perf_tot, perf_sim_list, paired_ag_pop_idx_list = r
                    
                    # average of perf of the i-th agent of first population paired with n other agents of the second population
                    first_half_performances[i] = perf_tot  
                    
                    for j, (perf_sim, paired_pop_idx) in \
                        enumerate(zip(perf_sim_list, paired_ag_pop_idx_list)):
                            # adding single sim performance to second agent
                            second_half_performances[j][paired_pop_idx[1]] = perf_sim 
                            # j is 0 for aligned agents, 1 if first agent is one step up wrt to
                            # second agent, ...

                # average performances on second population
                second_half_performances = \
                    np.mean(
                        second_half_performances, axis=0
                    )  
                performances = np.array([first_half_performances, second_half_performances])
                if self.num_pop == 1:
                    # joined the two half performances in one
                    performances = np.concatenate(performances)
            else:
                performances = np.array([p[0] for p in run_result])
        else:
            # num_pop > 2
            if self.num_cores > 1:
                # run parallel job            
                pop_results = Parallel(n_jobs=self.num_cores)(
                    delayed(sim_copy.run_pop_simulations)(populations, i, random_seed) \
                    for i in range(population_size)
                )
            else:
                # single core                
                pop_results = [
                    sim_copy.run_pop_simulations(populations, i, random_seed)
                    for i in range(population_size)
                ]

            # compute performance of each agent in population 
            # (paird with num_random_pairings other agents)
            pop_performances = np.zeros(
                (self.num_pop, self.num_random_pairings, population_size)
            )
            
            for i, pop_r in enumerate(pop_results):
                # pop_r contains the results from run_pop_simulations 
                # relative to agent at index i
                # pop_r is a list of num_pop - 1 tuples: 
                # (perf_tot, perf_sim_list, paired_ag_pop_idx_list)
                for p, r in enumerate(pop_r):
                    # p is the population_index of current pop_r
                    perf_tot, perf_sim_list, paired_ag_pop_idx_list = r
                    
                    for j, (perf_sim, paired_pop_idx) in \
                        enumerate(zip(perf_sim_list, paired_ag_pop_idx_list)):
                            # adding single sim performance to second agent
                            paired_pop, paired_i = paired_pop_idx
                            pop_performances[p][j+p][i] = perf_sim # current agent
                            assert paired_i == i
                            pop_performances[paired_pop][p][i] = perf_sim # paired agent

            # average performances in each population across pairs
            performances = np.mean(pop_performances, axis=1)  

        return performances

    def normalize_performance(self, performance):
        '''
        Returns the normalized performance
        such that best performance == 0
        '''
        return self.max_mean_distance - performance


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
    default_gen_structure = gen_structure.DEFAULT_GEN_STRUCTURE(1,2)
    rs = RandomState(3)
    run_result, _, data_record_list = get_simulation_data_from_random_agent(
        default_gen_structure, rs, num_dim=2)
    perf = run_result[0]
    print("Performance: ", perf)
    utils.save_json_numpy_data(data_record_list, 'data/simulation.json')


def ger_worst_performance(num_iter):
    worst = 0
    default_gen_structure = gen_structure.DEFAULT_GEN_STRUCTURE(1,2)    
    rs = RandomState(None)
    for _ in range(num_iter):
        run_result, _, _ = get_simulation_data_from_random_agent(default_gen_structure, rs)
        perf = run_result[0]
        if perf > worst:
            worst = perf
            print('Worst perf: ', worst)


if __name__ == "__main__":
    test_simulation()
    # ger_worst_performance(100)
