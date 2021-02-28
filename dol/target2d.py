"""
TODO: Missing module docstring
"""

import numpy as np
from dataclasses import dataclass, field
from numpy.random import RandomState
from dol.params import ENV_SIZE, HALF_ENV_SIZE

FOCAL_DISTANCE = HALF_ENV_SIZE
LEMNISCATE_A = FOCAL_DISTANCE * np.sqrt(2)

@dataclass
class Target2D:
    
    num_data_points: int
    num_trials: int
    rs: RandomState

    def __post_init__(self):
        if self.rs is None:
            self.trial_vel = np.repeat(np.arange(1,self.num_trials/2+1),2)[:self.num_trials] # 1, 1, 2, 2
            self.trial_vel[1::2] *= -1 # +, -, +, -, ...
            self.trial_start_phase = np.full(self.num_trials, 6*np.pi/4)
            self.trial_start_phase[1::2] *= -1 # +, -, +, -, ...
        else:
            max_vel = np.ceil(self.num_trials/2)
            # max_pos = ENV_SIZE/8
            self.trial_vel = self.rs.choice([-1,1]) * self.rs.uniform(1, max_vel, self.num_trials)
            self.trial_start_phase = self.rs.uniform(0, 2*np.pi, self.num_trials)

    def set_pos_vel(self, trial):
        # init start_phase        
        self.start_phase = self.trial_start_phase[trial]

        # init vel
        self.vel = self.trial_vel[trial]

    def compute_positions(self, trial):

        self.set_pos_vel(trial)
        
        # Lemniscate of Bernoulli function
        # using parametrix equation 
        # https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli
        # LEMNISCATE_A == FOCAL_DISTANCE * sqrt(2)
        t = 0.1 + self.start_phase + np.arange(self.num_data_points)/100 * self.vel
        x = LEMNISCATE_A * np.cos(t) / 1 + np.square(np.sin(t))
        y = LEMNISCATE_A * np.sin(t) * np.cos(t) / 1 + np.square(np.sin(t))

        self.positions = np.column_stack((x, y)) # x,y in two columns
        return self.positions

def test_target():
    t = Target2D(
        num_data_points = 500
    )    
    print(t.compute_positions(0))
    

if __name__ == "__main__":
    test_target()
