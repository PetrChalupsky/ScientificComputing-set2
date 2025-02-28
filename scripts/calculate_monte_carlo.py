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

width = 100
seed = 123
steps = 20000

final_clusters = []
simulations = []
p_values = [0.2, 0.4, 0.6, 0.8, 1.0]

for p in p_values:
    print("p:", p)
    all_grids = start_simulation(seed, steps, width, p)
    np.save(f"data/monte_carlo_final_cluster_{p}", all_grids[-1].copy())
    np.save(f"data/monte_carlo_all_grids_{p}", all_grids.copy())
