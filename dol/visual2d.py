"""
TODO: Missing module docstring
"""

import numpy as np
from numpy.random import RandomState
import pygame
from dol import gen_structure
from dol import tracker2d
from dol.tracker2d import Tracker2D, BODY_RADIUS, SENSOR_RADIUS, EYE_DIVERGENCE_LR_ANGLES
from dol.params import ENV_SIZE, HALF_ENV_SIZE, BODY_RADIUS, SENSOR_RADIUS

CANVAS_SIZE = int(2.5*ENV_SIZE)
ZOOM_FACTOR = 1.5
REFRESH_RATE = 50
SENSOR_RANGE_LINE_LENGTH = ZOOM_FACTOR * HALF_ENV_SIZE

CANVAS_CENTER = np.full(2, CANVAS_SIZE/2)

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
target_color = red
tracker_color = blue
target_tracker_vpos = [CANVAS_SIZE/3*2, CANVAS_SIZE/3]
tracker_sensor_color = yellow

body_radius = BODY_RADIUS * ZOOM_FACTOR
tracker_sensor_radius = SENSOR_RADIUS * ZOOM_FACTOR

class Visualization2D:

    def __init__(self, simulation):

        self.simulation = simulation

        pygame.init()

        self.main_surface = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE))


    def draw_target(self, target_pos):
        pos = ZOOM_FACTOR * target_pos + CANVAS_CENTER
        pygame.draw.circle(self.main_surface, target_color, pos, body_radius, width=0)

    def draw_tracker(self, tracker):
        tracker_pos = ZOOM_FACTOR * tracker.position + CANVAS_CENTER
        # print('tracker_pos', tracker_pos)
        pygame.draw.circle(self.main_surface, tracker_color, tracker_pos, body_radius, width=0)
        signals_strenght = tracker.signals_strength
        if not tracker2d.XY_MODE:
            abs_eyes_pos = tracker.get_abs_eyes_pos()
            eye_thetas = tracker.angle + EYE_DIVERGENCE_LR_ANGLES
            # draw eye of the agent
            for e in range(2):
                eye_sig = signals_strenght[e]
                eye_theta = eye_thetas[e]
                eyes_pos = ZOOM_FACTOR * abs_eyes_pos[e] + CANVAS_CENTER
                # print('eyes_pos', e+1, eyes_pos)
                pygame.draw.circle(self.main_surface, tracker_sensor_color, eyes_pos, tracker_sensor_radius)
                
                # draw sensors cones
                '''
                eye_ang_start = tracker.eyes_vision_angle_start[e]
                eye_ang_end = tracker.eyes_vision_angle_end[e]
                draw_line(self.main_surface, eyes_pos, eye_ang_start, SENSOR_RANGE_LINE_LENGTH, white)
                draw_line(self.main_surface, eyes_pos, eye_ang_end, SENSOR_RANGE_LINE_LENGTH, white)
                '''

                # draw eyes signals
                '''
                sign_line_length = ZOOM_FACTOR * HALF_ENV_SIZE
                sign_line = utils.linmap(eye_sig, (0., 1.), (0., HALF_ENV_SIZE))
                draw_line(self.main_surface, eyes_pos, eye_theta, sign_line, tracker_sensor_color)
                '''
            
            
    def start_simulation_from_data(self, trial_index, data_record):
        running = True

        clock = pygame.time.Clock()
        
        duration = self.simulation.num_data_points        

        target_positions = data_record['target_position'][trial_index]
        tracker_positions = data_record['tracker_position'][trial_index]
        tracker_angles = data_record['tracker_angle'][trial_index]
        tracker_signals_strength = data_record['tracker_signals'][trial_index]

        i = 0

        tracker = Tracker2D()
        tracker.init_params_trial(trial_index)

        while running and i<duration:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            # reset canvas
            self.main_surface.fill(black)

            target_pos = target_positions[i]
            tracker.set_position_and_angle_signals_strength(                
                tracker_positions[i], 
                tracker_angles[i],
                tracker_signals_strength[i]                
            )
            tracker.compute_signal_strength_and_delta_target(target_pos)

            # draw target and tracker            
            self.draw_target(target_pos)
            self.draw_tracker(tracker)

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


def draw_line(surface, x1y1, theta, length, color):
    x2y2 = (
        int(x1y1[0] + length * np.cos(theta)),
        int(x1y1[1] + length * np.sin(theta))
    )
    pygame.draw.line(surface, color, x1y1, x2y2, width=1)

    

def test_visual():
    from dol import simulation    
    run_result, sim, data_record_list = simulation.get_simulation_data_from_random_agent(
        gen_struct = gen_structure.DEFAULT_GEN_STRUCTURE(2),
        rs = RandomState(2),
        num_dim = 2,
    )
    vis = Visualization2D(sim)
    vis.start_simulation_from_data(trial_index=0, data_record=data_record_list[0])


if __name__ == "__main__":
    test_visual()
