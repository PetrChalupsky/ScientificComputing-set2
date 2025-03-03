import numpy as np
from numba import njit
import math
import warnings
import sys


@njit
def initialize_grid(width, create_object):
    """
    Initialize grid given a width as parameter. It assumes a square grid.
    The upper row is equal to 1. The rest of the grid is equal to 0.
    """
    # Set empty grid
    c = np.zeros((width, width))

    # Set upper and lower boundary conditions
    if create_object == True:
        return c

    c[width - 1, :] = 1

    return c


@njit
def create_objects(list_objects, width):
    """
    Create an object (sink) on a grid.
    """
    objects_grid = initialize_grid(width, True)
    for objects in list_objects:
        row_start, row_end, column_start, column_end = objects
        objects_grid[row_start:row_end, column_start:column_end] = 1
    return objects_grid


@njit
def sor_with_objects(width, eps, omega, objects, diffusion_grid):
    """
    Given the input makes an initial grid and updates it
    for a given time. Returns the final grid.
    """

    # Initialize the grid
    if diffusion_grid != None:
        new_grid = diffusion_grid
    else:
        new_grid = initialize_grid(width, False)

    # Update grid while difference larger than epsilon
    delta = 100
    delta_list = []
    k = 0
    while delta >= eps and k < 10000:
        grid = new_grid.copy()
        for i in range(1, width - 1):
            for j in range(width):
                if objects[i, j] == 1:
                    new_grid[i, j] = 0
                else:
                    new_grid[i, j] = (
                        0.25
                        * omega
                        * (
                            new_grid[(i + 1) % (width), j]
                            + new_grid[(i - 1) % (width), j]
                            + new_grid[i, (j - 1) % (width)]
                            + new_grid[i, (j + 1) % (width)]
                        )
                        + (1 - omega) * new_grid[i, j]
                    )

        delta = np.max(np.abs(new_grid - grid))
        delta_list.append(delta)
        k = k + 1

    return new_grid


def determine_spread(width, eta, diffusion_grid, current_object):
    """
    Determine the next cell where the cluster will grow and return the cluster.
    """
    # Find possible positions
    candidates = []
    # minus 2 as we want to preserve initial source - discuss with Bartek
    for i in range(width - 2):
        for j in range(width - 1):
            if current_object[i, j] == 0:
                if (
                    current_object[i - 1, j] == 1
                    or current_object[i + 1, j] == 1
                    or current_object[i, j + 1] == 1
                    or current_object[i, j - 1] == 1
                ):
                    candidates.append((i, j))

    tot_c = 0
    cum_concentration = []
    for index in candidates:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            concentration_candidate = (
                np.float64(diffusion_grid[index[0], index[1]]) ** eta
            )
        if math.isnan(concentration_candidate):
            concentration_candidate = 0
        tot_c += concentration_candidate
        cum_concentration.append(tot_c)
    if tot_c == 0:
        print("The concentration in the grid is zero")
        sys.exit()

    cum_concentration = np.array(cum_concentration) / tot_c
    p = np.random.rand()
    for i, concentration in enumerate(cum_concentration):
        if p < concentration:
            chosen_index = candidates[i]
            break

    current_object[chosen_index[0], chosen_index[1]] = 1

    return current_object


def run_DLA(width, eta, omega, cluster):
    """
    Run simple DLA.
    """
    eps = 0.00001
    diffusion_grid = None
    for _ in range(500):
        diffusion_grid = sor_with_objects(width, eps, omega, cluster, diffusion_grid)
        cluster = determine_spread(int(width), eta, diffusion_grid, cluster)

    cluster = np.array(cluster)

    return diffusion_grid, cluster
