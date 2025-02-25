"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Runs time dependendent diffusion for couple selected time intervals.
"""

import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from scientificcomputing_set2.monte_carlo import start_simulation

width = 4
seed = 123
steps = 20

all_grids = start_simulation(seed, steps, width)

# kan hier data saven
