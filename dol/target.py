"""
TODO: Missing module docstring
"""

import numpy as np
from dataclasses import dataclass, field
from numpy.random import RandomState

@dataclass
class Target:
    
    num_data_points: int
    trial_vel: list = None
    trial_start_pos: list = None
    trial_delta_bnd: list = None # margin from left and right boundaries

    def __post_init__(self):
        from dol.simulation import ENV_SIZE
        self.half_env_size = ENV_SIZE/2        

    def set_pos_vel(self, trial):
        # init pos        
        self.start_pos = self.trial_start_pos[trial]
        self.pos = self.start_pos

        # init vel
        self.start_vel = self.trial_vel[trial]
        self.vel = self.start_vel

    def get_next_boundary_pos(self, trial):        
        return self.half_env_size - self.trial_delta_bnd[trial]

    def print_start_pos_vel(self):
        print("Start position: ", self.start_pos)
        print("Velocity: ", self.start_vel)

    def compute_positions(self, trial):

        self.positions = np.zeros(self.num_data_points)        

        self.set_pos_vel(trial)
                        
        bnd_pos_abs = self.get_next_boundary_pos(trial)

        self.positions[0] = self.pos
        
        for i in range(1, self.num_data_points):                          
            self.pos += self.vel        
            pos_sign, pos_abs = np.sign(self.pos), np.abs(self.pos)                      
            self.positions[i] = self.pos      
            # bnd_pos_sign same as sign(vel)
            if pos_sign==np.sign(self.vel) and pos_abs >= bnd_pos_abs:
                overstep = pos_abs - bnd_pos_abs
                self.pos = pos_sign * (bnd_pos_abs - overstep) # go back by overstep
                self.vel = -self.vel # invert velocity            
                bnd_pos_abs = self.get_next_boundary_pos(trial)    
        return self.positions

def test_target_constant():
    t = Target(
        num_data_points = 500,
        trial_vel = [1],
        trial_start_pos = [0],
        trial_delta_bnd = [0],
    )    
    print(t.compute_positions(0))

def test_target_standard_trials():
    t = Target(        
        num_data_points = 500,
        trial_vel = [-1],
        trial_start_pos = [10],
        trial_delta_bnd = [5],
    )    
    print(t.compute_positions(0))

def test_max_distance():
    t = Target(        
        num_data_points = 500,
        trial_vel = [-2],
        trial_start_pos = [0],
        trial_delta_bnd = [0],
    )    
    target_pos = t.compute_positions(0)
    tracker_vel = 20
    tracker_pos = np.arange(500)*tracker_vel
    delta_mean = np.mean(np.abs(tracker_pos-target_pos))
    print('Max delta:', delta_mean)
    

if __name__ == "__main__":
    test_target_constant()
    test_target_standard_trials()
    test_max_distance()