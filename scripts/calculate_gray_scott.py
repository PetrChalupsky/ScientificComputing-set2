"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Calculates final concentrations of Gray-Scott reaction-diffusion model 
    using an explicit finite difference method.
"""

from scientificcomputing_set2.gray_scott import gray_scott, gray_scott_noise
import numpy as np

# Parameters
width = 128
dx = 1
dt = 1
Du = 0.16
Dv = 0.08
t = 5000
bc = 'periodic'

# Calculate concentrations without and with noise
f1, k1 = 0.02,0.05
grid_u1 = gray_scott(width, dx, dt, t, Du, Dv, f1, k1, bc)[0]
grid_u1_noise = gray_scott_noise(width, dx, dt, t, Du, Dv, f1, k1, bc)[0]
    
f2, k2 = 0.022, 0.051
grid_u2 = gray_scott(width, dx, dt, t, Du, Dv, f2, k2, bc)[0]
grid_u2_noise = gray_scott_noise(width, dx, dt, t, Du, Dv, f2, k2, bc)[0]
    
f3, k3 = 0.01, 0.033
grid_u3 = gray_scott(width, dx, dt, t, Du, Dv, f3, k3, bc)[0]
grid_u3_noise = gray_scott_noise(width, dx, dt, t, Du, Dv, f3, k3, bc)[0]

# Save data
np.save("data/gs_0.02_0.05_periodic_5000.npy", grid_u1)
np.save("data/gs_0.022_0.051_periodic_5000.npy", grid_u2)
np.save("data/gs_0.01_0.033_periodic_5000.npy", grid_u3)

np.save("data/gs_noise_0.02_0.05_periodic_5000.npy", grid_u1_noise)
np.save("data/gs_noise_0.022_0.051_periodic_5000.npy", grid_u2_noise)
np.save("data/gs_noise_0.01_0.033_periodic_5000.npy", grid_u3_noise)

    