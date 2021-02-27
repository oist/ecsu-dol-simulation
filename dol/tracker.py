"""
TODO: Missing module docstring
"""

from dataclasses import dataclass, field
import numpy as np
from dol.utils import linmap

@dataclass
class Tracker:

    # to be initialized via init_params    
    position: float = None 
    angle: float = None                 
    velocity: float = None     
    wheels: np.ndarray = None  
    signals_strength : np.ndarray = None
    
    def init_params(self):
        from dol.simulation import ENV_SIZE
        self.half_env_size = ENV_SIZE/2
        self.position = 0.
        self.angle = 0.
        self.velocity = 0.
        self.wheels = np.zeros(2)
        self.signals_strength = np.zeros(2)

    def move_one_step(self):
        self.velocity = self.wheels[1] - self.wheels[0]
        self.angle = 0 if self.velocity>0 else -np.pi
        self.position += self.velocity

    def compute_signal_strength_and_delta_target(self, target_position):
        self.delta_target = target_position - self.position
        delta_abs = np.abs(self.delta_target)
        if delta_abs <= 1:
            # consider tracker and target overlapping -> max signla left and right sensor
            self.signals_strength = np.ones(2)
        elif delta_abs >= self.half_env_size:
            # signals gos to zero if beyond half env_size
            self.signals_strength = np.zeros(2)
        else:
            signal_index = 1 if self.delta_target > 0 else 0 # right or left
            self.signals_strength = np.zeros(2)
            # signals_strength[signal_index] = 1/delta_abs
            # better if signal decreases linearly
            self.signals_strength[signal_index] = linmap(
                delta_abs, [1,self.half_env_size],[1,0])

def test_tracker():
    Tracker(0)

if __name__ == "__main__":
    test_tracker()