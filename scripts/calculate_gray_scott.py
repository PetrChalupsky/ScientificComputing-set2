"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Calculates concentrations of Gray-Scott reaction-diffusion model 
    using an explicit finite difference method.
"""

from scientificcomputing_set2.gray_scott import (
    gray_scott,
    gray_scott_noise,
    all_concentrations_noise,
)
import numpy as np

# Parameters
width = 128
dx = 1
dt = 1
Du = 0.16
Dv = 0.08
t = 5000
bc = "periodic"

# Calculate concentrations without and with noise

# Parameter values 1
f1, k1 = 0.02, 0.05
grid_u1, grid_v1, c_u1, c_v1 = gray_scott(width, dx, dt, t, Du, Dv, f1, k1, bc)
grid_u1_noise, grid_v1_noise, c_u1_noise, c_v1_noise = gray_scott_noise(
    width, dx, dt, t, Du, Dv, f1, k1, bc, seed=0
)
all_concentrations_u1, all_concentrations_v1 = all_concentrations_noise(
    width, dx, dt, t, Du, Dv, f1, k1, bc
)

# Parameter values 2
f2, k2 = 0.022, 0.051
grid_u2, grid_v2, c_u2, c_v2 = gray_scott(width, dx, dt, t, Du, Dv, f2, k2, bc)
grid_u2_noise, grid_v2_noise, c_u2_noise, c_v2_noise = gray_scott_noise(
    width, dx, dt, t, Du, Dv, f2, k2, bc, seed=0
)
all_concentrations_u2, all_concentrations_v2 = all_concentrations_noise(
    width, dx, dt, t, Du, Dv, f2, k2, bc
)

# Parameter values 3
f3, k3 = 0.035, 0.060
grid_u3, grid_v3, c_u3, c_v3 = gray_scott(width, dx, dt, t, Du, Dv, f3, k3, bc)
grid_u3_noise, grid_v3_noise, c_u3_noise, c_v3_noise = gray_scott_noise(
    width, dx, dt, t, Du, Dv, f3, k3, bc, seed=0
)
all_concentrations_u3, all_concentrations_v3 = all_concentrations_noise(
    width, dx, dt, t, Du, Dv, f3, k3, bc
)

# Save data
np.save("data/gs_0.02_0.05_periodic_5000.npy", grid_u1)
np.save("data/gs_0.022_0.051_periodic_5000.npy", grid_u2)
np.save("data/gs_0.035_0.060_periodic_5000.npy", grid_u3)

np.save("data/gs_noise_0.02_0.05_periodic_5000.npy", grid_u1_noise)
np.save("data/gs_noise_0.022_0.051_periodic_5000.npy", grid_u2_noise)
np.save("data/gs_noise_0.035_0.060_periodic_5000.npy", grid_u3_noise)

np.save("data/concentration_u_gs_0.02_0.05_periodic_5000.npy", c_u1)
np.save("data/concentration_v_gs_0.02_0.05_periodic_5000.npy", c_v1)

np.save("data/concentration_u_gs_0.022_0.051_periodic_5000.npy", c_u2)
np.save("data/concentration_v_gs_0.022_0.051_periodic_5000.npy", c_v2)

np.save("data/concentration_u_gs_0.035_0.060_periodic_5000.npy", c_u3)
np.save("data/concentration_v_gs_0.035_0.060_periodic_5000.npy", c_v3)

np.save("data/all_u_noise_gs_0.02_0.05_periodic_5000.npy", all_concentrations_u1)
np.save("data/all_v_noise_gs_0.02_0.05_periodic_5000.npy", all_concentrations_v1)

np.save("data/all_u_noise_gs_0.022_0.051_periodic_5000.npy", all_concentrations_u2)
np.save("data/all_v_noise_gs_0.022_0.051_periodic_5000.npy", all_concentrations_v2)

np.save("data/all_u_noise_gs_0.035_0.060_periodic_5000.npy", all_concentrations_u3)
np.save("data/all_v_noise_gs_0.035_0.060_periodic_5000.npy", all_concentrations_v3)
