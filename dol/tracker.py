"""
TODO: Missing module docstring
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Tracker:

    position: float = None     # to be initialized via init_params
    velocity: float = None     # to be initialized via init_params
    wheels: np.ndarray = None  # to be initialized via init_params
    
    def init_params(self):
        self.position = 0
        self.velocity = 0
        self.wheels = np.zeros(2)

    def move_one_step(self):
        self.velocity = np.diff(self.wheels) # wheels[1] - wheels[0]
        self.position += self.velocity

def test_tracker():
    Tracker(0)

if __name__ == "__main__":
    test_tracker()