"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Visualize speedup of DLA with SOR.
"""

import matplotlib.pyplot as plt
import numpy as np

cuda_times = np.load("data/run_cuda_DLA_times.npy")
parallel_cpu_times = np.load("data/run_parallel_cpu_DLA_times.npy")
times = np.load("data/run_DLA_times.npy")
widths = np.load("data/widths.npy")

plt.scatter(widths, times, label="non-parallelized")
plt.scatter(widths, parallel_cpu_times, label="parallel CPU")
plt.scatter(widths, cuda_times, label="GPU")
plt.xlabel("Grid width")
plt.ylabel("Wall time (s)")

plt.legend()
plt.savefig("results/speedup_DLA.png", dpi=300)
