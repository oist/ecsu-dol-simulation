"""
TODO: Missing module docstring
"""

from dataclasses import dataclass, field
import numpy as np
from dol.utils import linmap, modulo_radians

BODY_RADIUS = 4
SENSOR_RADIUS = 2

SENSORS_DIVERGENCE_ANGLE = np.pi/4 # angle between each sensor and axes of symmetry (angle of agent)
    

@dataclass
class Tracker2D:

    # to be initialized via init_params
    position: np.ndarray = None     
    velocity: float = None          
    wheels: np.ndarray = None       
    angle: float = None             
    sensors_angle: np.ndarray = None
    sensors_pos: np.ndarray = None
    signals_strength : np.ndarray = None

    def init_params(self):
        from dol.simulation import ENV_SIZE
        self.half_env_size = ENV_SIZE/2
        self.position = np.zeros(2)
        self.velocity = 0
        self.wheels = np.zeros(2)
        self.angle = 0
        self.signals_strength = np.zeros(2)
        self.__update_sensors_pos()

    def __update_sensors_pos(self):
        # sensors position are relative to center of the agents
        self.sensors_angle = np.array([
            modulo_radians(self.angle + SENSORS_DIVERGENCE_ANGLE),  # left sensor
            modulo_radians(self.angle - SENSORS_DIVERGENCE_ANGLE)   # right sensor
        ])
        # shape 2,2 (one row per sensor)
        self.sensors_pos = self.position + \
            BODY_RADIUS * \
            np.array([np.cos(self.sensors_angle), np.sin(self.sensors_angle)])
        
    def set_position_and_angle(self, pos, angle):
        self.position = pos # absolute position
        self.angle = angle
        self.__update_sensors_pos()

    def move_one_step(self):
        self.velocity = self.wheels[1] - self.wheels[0]  # right - left
        delta_angle = self.velocity / BODY_RADIUS
        self.angle += delta_angle
        avg_displacement = np.mean(self.wheels)
        delta_xy = avg_displacement * np.array([np.cos(self.angle), np.sin(self.angle)])
        self.position += delta_xy
        if delta_angle:
            self.__update_sensors_pos()

    def compute_signal_strength_and_delta_target(self, target_position):                
        self.delta_target = np.linalg.norm(
            target_position - self.position
        )
        
        # distance: 2 element vector
        abs_dists_sensors_target = np.linalg.norm(  
            self.sensors_pos - target_position, axis=0
        ) 
        
        # assert (abs_dists_sensors_target>=0).all()

        # signal decreases linearly with distance
        self.signals_strength = linmap(
            abs_dists_sensors_target, [1,self.half_env_size],[1,0]
        ) 

        # if either of the two signals is stronger than one make it 1
        # (overlapping tracker - target)
        self.signals_strength[self.signals_strength>1] = 1

        # if signals is negative make it 0
        # (beyond environment)
        self.signals_strength[self.signals_strength<0] = 0
        

def test_tracker():
    Tracker2D(0)

if __name__ == "__main__":
    test_tracker()