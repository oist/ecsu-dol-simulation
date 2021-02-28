"""
TODO: Missing module docstring
"""

import numpy as np
from dataclasses import dataclass, field
from numpy.random import RandomState


@dataclass
class Target2D:
    
    num_data_points: int
    trial_vel: list = None
    trial_start_pos: list = None

    def set_pos_vel(self, trial):
        # init pos        
        self.pos = self.trial_start_pos[trial]
        assert type(self.pos) is np.ndarray

        # init vel
        self.start_vel = self.trial_vel[trial]
        self.vel = self.start_vel

    def compute_positions(self, trial):

        self.set_pos_vel(trial)

        a = 150
        t = 0.1 + 6*np.pi/4 + np.arange(self.num_data_points)/70
        x = a * np.cos(t) / 1 + np.square(np.sin(t))
        y = a * np.sin(t) * np.cos(t) / 1 + np.square(np.sin(t))

        self.positions = np.column_stack((x, y)) # x,y in two columns
        return self.positions

def test_target():
    t = Target2D(
        num_data_points = 500,
        trial_vel = [1],
        trial_start_pos = [np.zeros(2)],
    )    
    print(t.compute_positions(0))
    

if __name__ == "__main__":
    test_target()
