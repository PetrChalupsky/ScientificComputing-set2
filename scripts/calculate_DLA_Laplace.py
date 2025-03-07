"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Runs the diffusion-limited aggregation simulation for different parameter values.
"""

from scientificcomputing_set2.DLA_Laplace import create_objects, run_DLA
import numpy as np

width = 100
list_objects = np.array([[2, 4, 49, 51]], dtype=np.int64)
cluster = create_objects(list_objects, int(width))


omega = 1.7
eta_list = [1, 0.2, 1.5]
for eta in eta_list:
    cluster = create_objects(list_objects, int(width))
    diffusion_grid_eta, cluster_eta = run_DLA(width, eta, omega, cluster)
    np.save(f"data/dla_cluster_eta{eta}", cluster_eta)
    np.save(f"data/dla_diffusion_eta{eta}", diffusion_grid_eta)
