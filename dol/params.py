"""
Main parameters for simulation experiment environment and agent body.
"""
import numpy as np
from pyevolver.evolution import MIN_SEARCH_VALUE, MAX_SEARCH_VALUE

ENV_SIZE = 400
HALF_ENV_SIZE = ENV_SIZE/2
BODY_RADIUS = 4
SENSOR_RADIUS = 2
EVOLVE_GENE_RANGE = (MIN_SEARCH_VALUE, MAX_SEARCH_VALUE)

# 2d
EYE_DIVERGENCE_ANGLE = np.radians(20)
EYE_VISION_ANGLE = np.radians(60)
