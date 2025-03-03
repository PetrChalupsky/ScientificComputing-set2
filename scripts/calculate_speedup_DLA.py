"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Measures amount time it takes to run a simulation given different implementations.
"""

from scientificcomputing_set2.DLA_Laplace import create_objects, run_DLA
from scientificcomputing_set2.parallel_DLA_Laplace import (
    run_cuda_DLA,
    run_parallel_cpu_DLA,
)
import numpy as np
import time


list_objects = np.array([[2, 4, 49, 51]], dtype=np.int64)

eta = 1
omega = 1.7
widths = [100, 200, 500, 1000]

run_DLA_times = []
run_cuda_DLA_times = []
run_parallel_cpu_DLA_times = []


for width in widths:
    start_time = time.time()
    cluster = create_objects(list_objects, int(width))
    diffusion_grid_eta, cluster_eta = run_DLA(width, eta, omega, cluster)
    total_time = time.time() - start_time
    print(f"Non-parallel, width = {width}: {total_time}")
    run_DLA_times.append(total_time)

for width in widths:
    start_time = time.time()
    cluster = create_objects(list_objects, int(width))
    diffusion_grid_eta, cluster_eta = run_cuda_DLA(width, eta, omega, cluster)
    total_time = time.time() - start_time
    print(f"GPU, width = {width}: {total_time}")
    run_cuda_DLA_times.append(total_time)

for width in widths:
    start_time = time.time()
    cluster = create_objects(list_objects, int(width))
    diffusion_grid_eta, cluster_eta = run_parallel_cpu_DLA(width, eta, omega, cluster)
    total_time = time.time() - start_time
    print(f"Parallel CPU, width = {width}: {total_time}")
    run_parallel_cpu_DLA_times.append(total_time)


np.save("data/run_DLA_times", run_DLA_times)
np.save("data/run_cuda_DLA_times", run_cuda_DLA_times)
np.save("data/run_parallel_cpu_DLA_times", run_parallel_cpu_DLA_times)
np.save("data/widths", widths)
