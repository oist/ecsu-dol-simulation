"""
TODO: Missing module docstring
"""

from dataclasses import dataclass, field
import numpy as np
from dol.utils import linmap, modulo_radians, angle_in_range
from dol.params import ENV_SIZE, HALF_ENV_SIZE, BODY_RADIUS, SENSOR_RADIUS, EYE_DIVERGENCE_ANGLE, EYE_VISION_ANGLE

# angle between left/right eye and axes of symmetry (angle of agent)
EYE_DIVERGENCE_LR_ANGLES = EYE_DIVERGENCE_ANGLE * np.array([1, -1]) 
# angle of vision of each eye
EYE_VISION_HALF_ANGLE = EYE_VISION_ANGLE / 2
DOUBLE_BODY_RADIUS = 2 * BODY_RADIUS

XY_MODE = True
COLLISION_MODE = False

@dataclass
class Tracker2D:

    def init_params_trial(self, trial_idx):
        self.position = np.zeros(2)
        self.velocity = 0
        self.wheels = np.zeros(2)
        if XY_MODE:
            self.angle = 0
        else:
            if trial_idx%2 ==0:
                self.angle = - np.pi/4 # 45 degree: face the target
            else:
                self.angle = 5*np.pi/4 # face the target
        self.signals_strength = np.zeros(4)
        self.collision = False     
        if not XY_MODE:   
            self.__update_eye_pos()

    def __update_eye_pos(self):
        # sensors position are relative to center of the agents        
        self.eyes_angle = modulo_radians(
            self.angle + EYE_DIVERGENCE_LR_ANGLES
        )

        # shape 2,2 (one row per sensor)
        self.eyes_pos = BODY_RADIUS * \
            np.array([np.cos(self.eyes_angle), np.sin(self.eyes_angle)]).transpose()
        
        self.eyes_vision_angle_start = modulo_radians(
            self.angle + EYE_DIVERGENCE_LR_ANGLES - EYE_VISION_HALF_ANGLE
        )        
        
        self.eyes_vision_angle_end = modulo_radians(
            self.eyes_vision_angle_start + EYE_VISION_ANGLE
        )
    
    def get_abs_eyes_pos(self):
        # get absolute positions of eyes
        return self.eyes_pos + self.position
        
    def set_position_and_angle_signals_strength(self, pos, angle, signals_strength):
        self.position = pos # absolute position
        self.angle = angle
        self.signals_strength = signals_strength
        if not XY_MODE: 
            self.__update_eye_pos()

    def move_one_step(self):  
        if XY_MODE:
            self.position += self.wheels
        else:
            self.velocity = self.wheels[1] - self.wheels[0]  # right - left
            delta_angle = self.velocity / BODY_RADIUS
            self.angle += delta_angle
            if self.collision:
                # reverse the angle and push him away for a distance = ENV_SIZE 
                self.angle -= np.pi
                avg_displacement = ENV_SIZE/4
            else:
                avg_displacement = np.mean(self.wheels)
            
            delta_xy = avg_displacement * np.array([np.cos(self.angle), np.sin(self.angle)])        
            self.position += delta_xy                    
            self.__update_eye_pos()

    def compute_signal_strength_and_delta_target(self, target_position, debug=False):                

        delta_tracker_target = target_position - self.position
        self.delta_target = np.linalg.norm(delta_tracker_target)

        if XY_MODE:
            self.signals_strength = np.zeros(4)
            delta_x, delta_y = delta_tracker_target
            abs_delta_x, abs_delta_y = abs(delta_x), abs(delta_y)
            if abs_delta_x <= HALF_ENV_SIZE:
                index = 0 if delta_x<0 else 1
                self.signals_strength[index] = linmap(abs_delta_x, [0, HALF_ENV_SIZE],[1,0])
            if abs_delta_y <= HALF_ENV_SIZE:
                index = 2 if delta_y<0 else 3
                self.signals_strength[index] = linmap(abs_delta_y, [0, HALF_ENV_SIZE],[1,0])
        
        else:

            if COLLISION_MODE:
                self.collision = self.delta_target < DOUBLE_BODY_RADIUS

            vector_eyes_target = target_position - self.get_abs_eyes_pos()
            
            # distance: 2 elements vector
            # we need to get the norm of each 2 vector row
            # always >= 0
            dists_sensors_target = np.linalg.norm(vector_eyes_target, axis=1)         
            
            angles_eyes_target = modulo_radians(
                np.arctan2(vector_eyes_target[:,1], vector_eyes_target[:,0])
            )
            
            # whether each eye sees the target
            eyes_see_target = np.array(
                [
                    angle_in_range(
                        angles_eyes_target[i],
                        self.eyes_vision_angle_start[i],
                        self.eyes_vision_angle_end[i]
                    )
                    for i in range(2)
                ]
            )
            # convert to 1/0
            eyes_see_target = 1. * eyes_see_target
            

            # signal decreases linearly with distance
            self.signals_strength = eyes_see_target * \
                linmap(
                    dists_sensors_target, [1, HALF_ENV_SIZE],[1,0]
                ) 
            

            # if either of the two signals is stronger than one make it 1
            # (overlapping tracker - target)
            self.signals_strength[self.signals_strength>1] = 1.

            # if signals is negative make it 0
            # (beyond environment)
            self.signals_strength[self.signals_strength<0] = 0.



    '''
    # old version
    def compute_signal_strength_and_delta_target(self, target_position):
        self.delta_target = np.linalg.norm(target_position - self.position)
        if self.delta_target <= 1:
            # consider tracker and target overlapping -> max signla left and right sensor
            self.signals_strength = np.ones(2)
        elif self.delta_target >= HALF_ENV_SIZE:
            # signals gos to zero if beyond half env_size
            self.signals_strength = np.zeros(2)
        else:
            signal_index = 1 if self.delta_target > 0 else 0 # right or left
            self.signals_strength = np.zeros(2)
            # signals_strength[signal_index] = 1/delta_abs
            # better if signal decreases linearly
            self.signals_strength[signal_index] = linmap(
                self.delta_target, [1,HALF_ENV_SIZE],[1,0])
    '''

def test_tracker():
    Tracker2D(0)

if __name__ == "__main__":
    test_tracker()