"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Solves Gray-Scott reaction-diffusion model using an explicit 
    finite difference method.
    
    Implementend boundary conditions:
    - periodic boundary conditions
    - Dirichlet boundary conditions: fixed value at boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def initialize_grids(width):
    """
    Initializes grids for both u and v concentrations given the width. 
    Everywhere in the system u = 0.5 is taken. For v a small square in 
    the center is equal to 0.25, while it is 0 elsewhere.

    Args:
        width: width of the grid

    Returns:
        Initialized grids for u and v
    """
    if width < 16:
        raise ValueError("Width must be greater or equal to 16")

    # Take u = 0.5 everywhere in the system
    grid_u = np.full((width, width), 0.5)
    
    # Take v = 0.25 in a small square in the center of the system, 
    # and 0 outside
    grid_v = np.zeros((width, width))
    
    left = width // 2 - 8 # Left/upper index of small square
    right = width // 2 + 8 # Right/lower index of small square

    # If width is uneven, make square larger by one 1 so that it can be centered
    if width%2 != 0:
        right = right + 1 

    grid_v[left:right, left:right] = 0.25
    
    return grid_u, grid_v

@njit
def update_grids_dirichlet(width, dx, dt, grid_u, grid_v, Du, Dv, f, k):
    """
    Updates grid according to explicit finite difference scheme 
    derived from the Gray-Scott model. With Dirichlet boundary
    conditions.

    Args:
        width: width of the grid
        dx: spatial step size
        dt: time step
        grid_u: matrix containing concentration of u at each cell
        grid_v: matrix containing concentration of v at each cell
        Du: diffusion constant u
        Dv: diffusion constant v
        f: rate at which u is supplied
        k: controls the rate at which v decays

    Returns:
        Updated grid for u and v
    """

    new_grid_u = grid_u.copy()
    new_grid_v = grid_v.copy()

    # For each cell calculate new value with the explicit scheme.
    for i in range(1, width-1):
        for j in range(1, width-1):

            # Diffusion term of u
            diffusion_u = Du * (grid_u[i + 1, j]
                + grid_u[i - 1, j]
                + grid_u[i, j - 1]
                + grid_u[i, j + 1]
                - 4*grid_u[i, j]) / dx**2
            
            # Update grid of u
            new_grid_u[i, j] = grid_u[i, j] + dt *(
                diffusion_u - grid_u[i, j]*grid_v[i, j]**2 + f*(1-grid_u[i,j]))
            
            # Diffusion term of v
            diffusion_v = Dv * (grid_v[i + 1, j]
                + grid_v[i - 1, j]
                + grid_v[i, j - 1]
                + grid_v[i, j + 1]
                - 4*grid_v[i, j]) / dx**2
            
            # Update grid of v
            new_grid_v[i, j] = grid_v[i, j] + dt*(
                diffusion_v + grid_u[i, j]*grid_v[i, j]**2 - (f+k)*grid_v[i,j])

    return new_grid_u, new_grid_v

@njit
def update_grids_periodic(width, dx, dt, grid_u, grid_v, Du, Dv, f, k):
    """
    Updates grid according to explicit finite difference scheme 
    derived from the Gray-Scott model. With periodic boundary
    conditions.

    Args:
        width: width of the grid
        dx: spatial step size
        dt: time step
        grid_u: matrix containing concentration of u at each cell
        grid_v: matrix containing concentration of v at each cell
        Du: diffusion constant u
        Dv: diffusion constant v
        f: rate at which u is supplied
        k: controls the rate at which v decays

    Returns:
        Updated grid for u and v
    """
    new_grid_u = grid_u.copy()
    new_grid_v = grid_v.copy()

    # For each cell calculate new value with the explicit scheme.
    for i in range(0, width):
        for j in range(0, width):

            # Diffusion term of u
            diffusion_u = Du * (grid_u[(i + 1)%width, j]
                + grid_u[(i - 1)%width, j]
                + grid_u[i, (j - 1)%width]
                + grid_u[i, (j + 1)%width]
                - 4*grid_u[i, j]) / dx**2
            
            # Update grid of u
            new_grid_u[i, j] = grid_u[i, j] + dt *(
                diffusion_u - grid_u[i, j]*grid_v[i, j]**2 + f*(1-grid_u[i,j]))
            
            # Diffusion term of v
            diffusion_v = Dv * (grid_v[(i + 1)%width, j]
                + grid_v[(i - 1)%width, j]
                + grid_v[i, (j - 1)%width]
                + grid_v[i, (j + 1)%width]
                - 4*grid_v[i, j]) / dx**2
            
            # Update grid of v
            new_grid_v[i, j] = grid_v[i, j] + dt*(
                diffusion_v + grid_u[i, j]*grid_v[i, j]**2 - (f+k)*grid_v[i,j])
            
    return new_grid_u, new_grid_v



@njit
def gray_scott(width, dx, dt, t, Du, Dv, f, k, bc):
    """
    Makes an initial grid and updates this for a given time. 
    Returns final grid.

    Args:
        width: width of the grid
        dx: spatial step size
        dt: time step
        t: final time
        Du: diffusion constant u
        Dv: diffusion constant v
        f: rate at which u is supplied
        k: controls the rate at which v decays
        bc: choice of boundary conditions, either 'periodic' or 'dirichlet'

    Returns:
        Final grid for u and v and list of their concentrations
    """

    # Number of timesteps
    steps = int(t / dt)

    # Initialize the grids
    grid_u, grid_v = initialize_grids(width)

    concentration_u = []
    concentration_v = []

    # Update the grid for calculated amount of steps
    step = 0

    # Periodic boundary conditions are chosen
    if bc == 'periodic':
        for step in range(steps):
            grid_u, grid_v = update_grids_periodic(width, dx, dt, grid_u, grid_v, Du, Dv, f, k)
            
            # Calculate total concentrations at each time step
            concentration_u.append(np.sum(grid_u))
            concentration_v.append(np.sum(grid_v))

            step = step + 1

        return grid_u, grid_v, concentration_u, concentration_v
    
    # Dirichlet boundary conditions are chosen
    if bc == 'dirichlet':
        for step in range(steps):
            grid_u, grid_v = update_grids_dirichlet(width, dx, dt, grid_u, grid_v, Du, Dv, f, k)
            
            # Calculate total concentrations at each time step
            concentration_u.append(np.sum(grid_u))
            concentration_v.append(np.sum(grid_v))

            step = step + 1

        return grid_u, grid_v, concentration_u, concentration_v

@njit
def gray_scott_noise(width, dx, dt, t, Du, Dv, f, k, bc, seed):
    """
    Makes an initial grid with noise and updates this for a given time. 
    Returns final grid.

    Args:
        width: width of the grid
        dx: spatial step size
        dt: time step
        t: final time
        Du: diffusion constant u
        Dv: diffusion constant v
        f: rate at which u is supplied
        k: controls the rate at which v decays
        bc: choice of boundary conditions, either 'periodic' or 'dirichlet'
        seed: random seed

    Returns:
        Final grid for u and v and list of their concentrations
    """

    # Number of timesteps
    steps = int(t / dt)

    # Initialize the grids
    np.random.seed(seed)

    # Random matrix containing uniformly distributed values between 0 and 0.1
    random_matrix = np.random.uniform(0, 0.1, size=(width, width))
    
    grid_u = initialize_grids(width)[0]
    grid_v = initialize_grids(width)[1] + random_matrix

    concentration_u = []
    concentration_v = []

    # Update the grid for calculated amount of steps
    step = 0

    # Periodic boundary conditions are chosen
    if bc == 'periodic':
        for step in range(steps):
            grid_u, grid_v = update_grids_periodic(width, dx, dt, grid_u, grid_v, Du, Dv, f, k)
            
            # Calculate total concentrations at each time step
            concentration_u.append(np.sum(grid_u))
            concentration_v.append(np.sum(grid_v))

            step = step + 1

        return grid_u, grid_v, concentration_u, concentration_v
    
    # Dirichlet boundary conditions are chosen
    if bc == 'dirichlet':
        for step in range(steps):
            grid_u, grid_v = update_grids_dirichlet(width, dx, dt, grid_u, grid_v, Du, Dv, f, k)
            
            # Calculate total concentrations at each time step
            concentration_u.append(np.sum(grid_u))
            concentration_v.append(np.sum(grid_v))

            step = step + 1

        return grid_u, grid_v, concentration_u, concentration_v
    
def all_concentrations_noise(width, dx, dt, t, Du, Dv, f, k, bc):
    """
    Calculates the total concentrations at each time step for multiple
    seeds.

    Args:
        width: width of the grid
        dx: spatial step size
        dt: time step
        t: final time
        Du: diffusion constant u
        Dv: diffusion constant v
        f: rate at which u is supplied
        k: controls the rate at which v decays
        bc: choice of boundary conditions, either 'periodic' or 'dirichlet'

    Returns:
        Arrays for u and v containing for each seed another array with 
        the total concentrations at each time step.
    """
    all_concentrations_u = []
    all_concentrations_v = []

    for seed in range(10):
        # Given a seed calculate concentration of u and add to the collecting array
        concentration_u = gray_scott_noise(width, dx, dt, t, Du, Dv, f, k, bc, seed)[2]
        all_concentrations_u.append(concentration_u)

        # Given a seed calculate concentration of v and add to the collecting array
        concentration_v = gray_scott_noise(width, dx, dt, t, Du, Dv, f, k, bc, seed)[3]
        all_concentrations_v.append(concentration_v)
    
    return all_concentrations_u, all_concentrations_v

