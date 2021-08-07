"""
Implements 1D animation of experimental simulation.
"""

import numpy as np
from numpy.random import RandomState
from dataclasses import dataclass
import pygame
from dol import utils
from dol.target import Target
from dol.phidget.slider import Slider    

ENV_WITH = 400
CANVAS_WIDTH = 800
CANVAS_HIGHT = 300
REFRESH_RATE = 50
SLIDER_GAIN = 8
BLOCKING_BOUNDARIES = True

HALF_ENV_WITH = ENV_WITH/2
ZOOM_FACTOR = CANVAS_WIDTH/ENV_WITH
CANVAS_H_CENTER = CANVAS_WIDTH/2

black = (0, 0, 0)
white = (255, 255, 255)
red = (190, 18, 27)
blue = (25, 82, 184)
green = (82, 240, 63)
yellow = (228, 213, 29)
target_tracker_color = [red, blue]
target_tracker_vpos = [CANVAS_HIGHT/3*2, CANVAS_HIGHT/3]
body_radius = 2 * ZOOM_FACTOR

@dataclass
class Visualization:

    num_people: int = 1
    trial_num: int = 1
    motor_control_mode: str = None
    exclusive_motors_threshold: float = None
    random_target_seed: int = None

    def __post_init__(self):

        self.__check_params__()

        pygame.init()

        self.main_surface = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HIGHT))

        self.num_sliders = 2 * self.num_people

        self.sliders = [
            Slider(port=i)
            for i in range(self.num_sliders)
        ]

        self.trial_idx = self.trial_num - 1        
        self.init_target_positions()
        self.duration = len(self.target_positions)
        self.target_h_pos = 0
        self.tracker_h_pos = 0

    def __check_params__(self):
        assert self.num_people==2 or self.motor_control_mode in [None, 'SWITCH'], \
            "With one agent motor_control_mode must be None or SWITCH"

        assert self.num_people==1 or self.motor_control_mode!=None, \
            "With two agents motor_control_mode must not be None"


    def init_target_positions(self):
        rs = None if self.random_target_seed is None else RandomState(self.random_target_seed)
        target = Target(num_data_points=2000, num_trials=4, rs=rs)    
        target.compute_positions(trial=self.trial_num-1)
        self.target_positions = target.positions


    def draw_target_tracker(self):

        for i,h_pos in enumerate([self.target_h_pos, self.tracker_h_pos]):
            color = target_tracker_color[i]
            v_pos = target_tracker_vpos[i]
            h_pos = ZOOM_FACTOR * h_pos + CANVAS_H_CENTER            
            radius = ZOOM_FACTOR * body_radius
            pos = [h_pos, v_pos]
            line_start_pos = [0, v_pos]
            line_end_pos = [CANVAS_WIDTH, v_pos]
            pygame.draw.line(self.main_surface, white, line_start_pos, line_end_pos, width=1)
            pygame.draw.circle(self.main_surface, color, pos, radius, width=0)
    
    def compute_next_tracker_pos(self):
        sliders_val = SLIDER_GAIN * np.array([
            1 - s.sensorValue
            for s in self.sliders
        ])

        switch_motors = self.motor_control_mode == 'SWITCH' and self.trial_idx % 2 != 0

        if self.num_people==1:
            # isolation mode
            if switch_motors:            
                motors = np.take(sliders_val, [1,0])
            else:
                motors = sliders_val  
        else: 
            # 2 people
            if self.motor_control_mode == 'SEPARATE':
                motors = np.take(sliders_val, [0,3])
            elif self.motor_control_mode == 'SWITCH':
                if switch_motors:
                    motors = np.take(sliders_val, [1,2])
                else:
                    motors = np.take(sliders_val, [0,3])
            elif self.motor_control_mode == 'OVERLAP':
                motors = 0.5 * \
                    (
                        np.take(sliders_val, [0,1]) +
                        np.take(sliders_val, [2,3])
                    )

        if self.exclusive_motors_threshold is not None and \
            np.all(motors>self.exclusive_motors_threshold):
            velocity = 0    
        else:
            velocity = np.diff(motors)

        self.tracker_h_pos += velocity

        if BLOCKING_BOUNDARIES:
            if self.tracker_h_pos > HALF_ENV_WITH:
                self.tracker_h_pos = HALF_ENV_WITH
            elif self.tracker_h_pos < -HALF_ENV_WITH:
                self.tracker_h_pos = -HALF_ENV_WITH

    def start(self):
        running = True

        clock = pygame.time.Clock()            

        i = 0

        while running and i<self.duration:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # reset canvas
            self.main_surface.fill(black)

            self.target_h_pos = self.target_positions[i]
            self.compute_next_tracker_pos()            

            # draw target and tracker            
            self.draw_target_tracker()

            # final traformations
            self.final_tranform_main_surface()
            pygame.display.update()

            clock.tick(REFRESH_RATE)

            i += 1


    def final_tranform_main_surface(self):
        '''
        final transformations:
        - shift coordinates to conventional x=0, y=0 in bottom left corner
        - zoom...
        '''
        self.main_surface.blit(pygame.transform.flip(self.main_surface, False, True), dest=(0, 0))

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Human simulation'
    )

    parser.add_argument('--num_people', type=int, default=1, choices=[1,2],
                        help='How many people in the experiment')
    parser.add_argument('--trial_num', type=int, default=1, 
                        help='Run a specific trial (e.g., for target behaviour)')
    parser.add_argument('--motor_control_mode', type=str, default=None,
                        choices=[None, 'SEPARATE', 'SWITCH', 'OVERLAP'],
                        help=
                        'Type of motor control'
                        'None: not applicable (if single agent)'
                        'SEPARATE: across trials the first agent always control the left motor and the second the right'
                        'SWITCH: the two agents switch control of L/R motors in different trials'
                        'OVERLAP: both agents control L/R motors (for a factor o half)') 
    parser.add_argument('--excl_motors', type=float, default=None,
                        help='prevent motors to run at the same time')    
    parser.add_argument('--random_target_seed', type=int,
                        help='Seed to re-run simulation with random target (None to obtain same results)')

    args = parser.parse_args()

    vis = Visualization(
        num_people=args.num_people,
        trial_num=args.trial_num,
        motor_control_mode=args.motor_control_mode,
        exclusive_motors_threshold=args.excl_motors,
        random_target_seed=args.random_target_seed,        
    )
    vis.start()

