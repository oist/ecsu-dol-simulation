"""
TODO: Missing module docstring
"""

import numpy as np
from numpy.random import RandomState
import pygame
from pyevolver.evolution import Evolution
from dol.simulation import Simulation
from dol import gen_structure
from dol import utils
from dol import target2d
from dol.tracker2d import Tracker2D, BODY_RADIUS, SENSOR_RADIUS

CANVAS_SIZE = 800
ZOOM_FACTOR = 2
REFRESH_RATE = 50

CANVAS_CENTER = np.full(2, CANVAS_SIZE/2)

SHIFT_CENTER_TO_TARGET = False

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


    def draw_target(self, target_pos, center_shift=0):
        pos = ZOOM_FACTOR * target_pos - ZOOM_FACTOR * center_shift + CANVAS_CENTER
        pygame.draw.circle(self.main_surface, target_color, pos, body_radius, width=0)

    def draw_tracker(self, tracker, center_shift=0):
        pos = ZOOM_FACTOR * tracker.position - ZOOM_FACTOR * center_shift + CANVAS_CENTER
        pygame.draw.circle(self.main_surface, tracker_color, pos, body_radius, width=0)
        # draw eye of the agent
        for sp in tracker.sensors_pos:
            sp = ZOOM_FACTOR * sp - ZOOM_FACTOR * center_shift + CANVAS_CENTER
            pygame.draw.circle(self.main_surface, tracker_sensor_color, sp, tracker_sensor_radius)
        

    
    def start_simulation_from_data(self, trial_index, data_record):
        running = True

        clock = pygame.time.Clock()
        
        duration = self.simulation.num_data_points        

        target_positions = data_record['target_position'][trial_index]
        tracker_positions = data_record['tracker_position'][trial_index]
        tracker_angles = data_record['tracker_angle'][trial_index]

        i = 0

        tracker = Tracker2D()

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
            tracker.set_position_and_angle(tracker_positions[i], tracker_angles[i])

            center_shift = target_pos[i] if SHIFT_CENTER_TO_TARGET else 0

            # draw target and tracker            
            self.draw_target(target_pos, center_shift)
            self.draw_tracker(tracker, center_shift)

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


def draw_line(surface, x1y1, theta, length):
    x2y2 = (
        int(x1y1[0] + length * np.cos(theta)),
        int(x1y1[1] + length * np.sin(theta))
    )
    pygame.draw.line(surface, white, x1y1, x2y2, width=1)

    

def test_visual():
    from dol import simulation    
    run_result, sim, data_record_list = simulation.get_simulation_data_from_random_agent(
        gen_struct = gen_structure.DEFAULT_GEN_STRUCTURE(2),
        rs = RandomState(None),
        num_dim = 2
    )
    vis = Visualization2D(sim)
    vis.start_simulation_from_data(trial_index=0, data_record=data_record_list[0])


if __name__ == "__main__":
    test_visual()
