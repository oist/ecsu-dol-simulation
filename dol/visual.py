"""
Implements 1D animation of experimental simulation.
"""

import numpy as np
from numpy.random import RandomState
import pygame
from dol.simulation import Simulation
from dol import gen_structure
from dol import utils
from pyevolver.evolution import Evolution

CANVAS_WIDTH = 800
CANVAS_HIGHT = 300
ZOOM_FACTOR = 2
REFRESH_RATE = 50

CANVAS_H_CENTER = CANVAS_WIDTH/2
SHIFT_CENTER_TO_TARGET = False

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
target_tracker_color = [red, blue]
target_tracker_vpos = [CANVAS_HIGHT/3*2, CANVAS_HIGHT/3]
body_radius = 2 * ZOOM_FACTOR

class Visualization:

    def __init__(self, simulation):

        self.simulation = simulation

        pygame.init()

        self.main_surface = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HIGHT))


    def draw_target_tracker(self, target_h_pos, tracker_h_pos, center_shift):

        for i,h_pos in enumerate([target_h_pos, tracker_h_pos]):
            color = target_tracker_color[i]
            v_pos = target_tracker_vpos[i]
            h_pos = ZOOM_FACTOR * h_pos - ZOOM_FACTOR * center_shift + CANVAS_H_CENTER            
            radius = ZOOM_FACTOR * body_radius
            pos = [h_pos, v_pos]
            line_start_pos = [0, v_pos]
            line_end_pos = [CANVAS_WIDTH, v_pos]
            pygame.draw.line(self.main_surface, white, line_start_pos, line_end_pos, width=1)
            pygame.draw.circle(self.main_surface, color, pos, radius, width=0)
        

    
    def start_simulation_from_data(self, trial_index, data_record):
        running = True

        clock = pygame.time.Clock()
        
        duration = self.simulation.num_data_points        

        target_h_positions = data_record['target_position'][trial_index]
        tracker_h_positions = data_record['tracker_position'][trial_index]

        i = 0

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

            target_h_pos = target_h_positions[i]
            tracker_h_pos = tracker_h_positions[i]

            center_shift = target_h_pos[i] if SHIFT_CENTER_TO_TARGET else 0

            # draw target and tracker            
            self.draw_target_tracker(target_h_pos, tracker_h_pos, center_shift)

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
        gen_struct = gen_structure.DEFAULT_GEN_STRUCTURE(1,2),
        rs = RandomState(None)
    )
    vis = Visualization(sim)
    vis.start_simulation_from_data(trial_index=0, data_record=data_record_list[0])


if __name__ == "__main__":
    test_visual()
