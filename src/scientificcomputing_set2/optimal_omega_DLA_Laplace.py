import numpy as np
from numba import njit
import math
import warnings
from scientificcomputing_set2.DLA_Laplace import sor_with_objects

class ConcentrationZero(Exception):
    pass


def determine_spread_search_omega(width, eta, diffusion_grid, current_object):
    """
    Determine the next cell where the cluster will grow and return the cluster.
    """
    # Find possible positions
    candidates = []
    # minus 2 as we want to preserve initial source - discuss with Bartek
    for i in range(width-2):
        for j in range(width-1):
            if current_object[i,j] == 0:
                if current_object[i-1,j] == 1 or current_object[i+1,j] == 1 or current_object[i, j+1] == 1 or current_object[i, j-1] == 1:
                    candidates.append((i,j))
  
    # Determine weights of each growth candidate
    tot_c = 0
    cum_concentration = []
    for index in candidates:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            concentration_candidate = np.float64(diffusion_grid[index[0],index[1]])**eta
        if math.isnan(concentration_candidate):
            concentration_candidate = 0
        tot_c += concentration_candidate
        cum_concentration.append(tot_c)
    if tot_c == 0:
        print('The concentration in the grid is zero')
        raise ConcentrationZero()

    cum_concentration = np.array(cum_concentration)/tot_c

    # Select the growth candidate
    p = np.random.rand()
    for i, concentration in enumerate(cum_concentration):
        if p < concentration:
            chosen_index = candidates[i]
            break

    

    current_object[chosen_index[0], chosen_index[1]] = 1

    return current_object

def search_omega_run_DLA(eta, omega, cluster):
    """
    Returns the number of iteration needed for given parameter.
    """
    width = 100
    eps = 0.00001
    diffusion_grid = None
    k = 0
    for i in range(500):
        try:
            k_temp, diffusion_grid = sor_with_objects(width, eps, omega, cluster, diffusion_grid)
            cluster = determine_spread_search_omega(int(width),eta, diffusion_grid, cluster)
            k += k_temp 
        except ConcentrationZero:
            print(f'Exiting early at iteartion {i}')
            return k



    cluster = np.array(cluster)

    return int(k)
