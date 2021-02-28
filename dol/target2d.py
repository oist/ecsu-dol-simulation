"""
TODO: Missing module docstring
"""

import numpy as np
from dataclasses import dataclass, field
from numpy.random import RandomState
from dol.params import HALF_ENV_SIZE

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
            # random target2D
            assert False, 'To be implemented'

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
        num_data_points = 500,
        trial_vel = [1],
        trial_start_phase = [np.zeros(2)],
    )    
    print(t.compute_positions(0))
    

if __name__ == "__main__":
    test_target()
