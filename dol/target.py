"""
TODO: Missing module docstring
"""

import numpy as np
from dataclasses import dataclass, field
from numpy.random import RandomState


@dataclass
class Target:
    
    num_data_points: int
    env_width: int = 400    
    trial_vel: list = None
    trial_start_pos: list = None
    trial_delta_bnd: list = None
    random_pos_max_value: float = None # upper bnd of start self.pos and delta bnd
    random_vel_max_value: float = None # upper bnd of start self.pos and delta bnd

    def __post_init__(self):
        assert (self.trial_start_pos==None) != (self.random_pos_max_value==None), \
            'trial_start_pos and random_pos_max_value cannot be both None or not None' 
        assert (self.trial_delta_bnd==None) != (self.random_pos_max_value==None), \
            'trial_delta_bnd and random_pos_max_value cannot be both None or not None' 
        assert (self.trial_vel==None) != (self.random_vel_max_value==None), \
            'trial_vel and random_vel_max_value cannot be both None or not None' 

        self.half_env_width = self.env_width/2
        

    def set_pos_vel(self, trial, rs):
        # init pos        
        if self.random_pos_max_value is None:
            self.start_pos = self.trial_start_pos[trial]
        else:
           self.start_pos = rs.choice([-1,1]) * self.random_pos_max_value * rs.random()
        self.pos = self.start_pos

        # init vel
        if self.random_vel_max_value is None:
            self.start_vel = self.trial_vel[trial]
        else:
            self.start_vel = rs.choice([-1,1]) * self.random_vel_max_value * rs.random()
        self.vel = self.start_vel

    def get_next_boundary_pos(self, trial, rs):        
        if self.random_pos_max_value is None:
            bnd_pos_abs = self.half_env_width - self.trial_delta_bnd[trial]
        else:
             bnd_pos_abs = self.half_env_width - rs.random()*self.random_pos_max_value
        return bnd_pos_abs

    def print_start_pos_vel(self):
        print("Start position: ", self.start_pos)
        print("Velocity: ", self.start_vel)

    def compute_positions(self, trial, rs):

        self.positions = np.zeros(self.num_data_points)        

        self.set_pos_vel(trial, rs)
                        
        bnd_pos_abs = self.get_next_boundary_pos(trial, rs)

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
                bnd_pos_abs = self.get_next_boundary_pos(trial, rs)    
        return self.positions

def test_target_constant():
    t = Target(
        num_data_points = 500,
        env_width = 100,
        trial_vel = [1],
        trial_start_pos = [0],
        trial_delta_bnd = [0],
    )    
    print(t.compute_positions(0, rs=None))

def test_target_standard_trials():
    t = Target(        
        num_data_points = 500,
        env_width = 100,
        trial_vel = [-1],
        trial_start_pos = [10],
        trial_delta_bnd = [5],
    )    
    print(t.compute_positions(0, rs=None))

def test_target_random_pos_vel():
    t = Target(      
        num_data_points = 500,  
        env_width = 100,
        random_pos_max_value = 20, # upper bnd of start self.pos and delta bnd
        random_vel_max_value = 5, # upper bnd of start self.pos and delta bnd
    )    
    rs=RandomState(0)
    print(t.compute_positions(0, rs))
    t.print_start_pos_vel()

if __name__ == "__main__":
    # test_target_constant()
    # test_target_standard_trials()
    test_target_random_pos_vel()