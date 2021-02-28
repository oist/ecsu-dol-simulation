"""
TODO: Missing module docstring
"""

import numpy as np
from dataclasses import dataclass, field
from numpy.random import RandomState
from dol.params import ENV_SIZE, HALF_ENV_SIZE

@dataclass
class Target:
    
    num_data_points: int
    num_trials: int
    rs: RandomState

    def __post_init__(self):
        if self.rs is None:
            self.trial_vel = np.repeat(np.arange(1,self.num_trials/2+1),2)[:self.num_trials] # 1, 1, 2, 2
            self.trial_vel[1::2] *= -1 # +, -, +, -, ...
            self.trial_start_pos = [0] * self.num_trials
            self.trial_delta_bnd = [0] * self.num_trials
        else:
            # random target
            max_vel = np.ceil(self.num_trials/2)
            max_pos = ENV_SIZE/8
            self.trial_vel = self.rs.choice([-1,1]) * self.rs.uniform(1, max_vel, self.num_trials),
            self.trial_start_pos = self.rs.uniform(-max_pos, max_pos, self.num_trials)
            self.trial_delta_bnd = self.rs.uniform(0, max_pos, self.num_trials)

    def set_pos_vel(self, trial):
        # init pos        
        self.start_pos = self.trial_start_pos[trial]
        self.pos = self.start_pos

        # init vel
        self.start_vel = self.trial_vel[trial]
        self.vel = self.start_vel

    def get_next_boundary_pos(self, trial):        
        return HALF_ENV_SIZE - self.trial_delta_bnd[trial]

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

def test_target():
    t = Target(
        num_data_points = 500
    )    
    print(t.compute_positions(0))

    

if __name__ == "__main__":
    test_target()
