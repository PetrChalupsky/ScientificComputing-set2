"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Monte Carlo implementation of the DLA, using random walkers.
    A cell has the value 0 when empty, 1 when containing the cluster and 2 when containing a walker.
"""

import numpy as np

global width


def initialize_grid(seed):
    """
    Initialize grid given a width as parameter. It assumes a square grid.
    A randomly selected cell on the bottom row is the starting cluster.
    """


    # Set empty grid
    grid = np.zeros((WIDTH, WIDTH))

    # Initialize the cluster
    np.random.seed(seed)
    random_col = np.random.randint(0, WIDTH)
    grid[-1, random_col] = 1

    return grid


def add_walker(grid):
    """
    Adds a Random Walker to the grid on the top boundary.
    """
    empty_cells = np.where(grid[0] == 0)[0]

    if empty_cells.size > 0:
        random_col = np.random.choice(empty_cells)
        grid[0, random_col] = 2

    return grid


def check_neighbourhood(grid, i_n, j_n):
    """
    Checks wheter the random walker is in the neightbourhood of the cluster.
    """
    if i_n + 1 < WIDTH and grid[i_n + 1, j_n] == 1:
        return True
    if i_n - 1 >= 0 and grid[i_n - 1, j_n] == 1:
        return True
    if j_n + 1 < WIDTH and grid[i_n, j_n + 1] == 1:
        return True
    if j_n - 1 >= 0 and grid[i_n, j_n - 1] == 1:
        return True

    return False


def new_indexes(temp_grid, grid, i, j):
    """
    Computes new indexes for random walker.

    Option1:
        Random walker goes up or down, if it goes out of bounds,
        it is removed and a new random walker is created.
    Option2:
        Random walker goes left or right, if it goes out of bounds,
        it reapears on the other side of the grid.
    """
    direction = np.random.choice(["up", "down", "left", "right"])

    i_n, j_n = i, j
    removed = False

    if direction == "up":
        if i > 0:
            i_n -= 1
        else:
            # temp_grid[i, j] = 0
            # temp_grid = add_walker(temp_grid)
            i_n -= 1
            removed = True
    elif direction == "down":
        if i < WIDTH - 1:
            i_n += 1
        else:
            i_n += 1
            removed = True
            # temp_grid[i, j] = 0
            # temp_grid = add_walker(temp_grid)
    elif direction == "left":
        if j > 0:
            j_n -= 1
        else:
            if grid[i_n, WIDTH - 1] != 1 and grid[i_n, WIDTH - 1] != 2: # Check the cell is unocupied
                j_n = WIDTH - 1
    elif direction == "right":
        if j < WIDTH - 1:
            j_n += 1
        else:
            if grid[i_n, 0] != 1 and grid[i_n, 0] != 2: # Check the cell is unocupied
                j_n = 0

    # Check if the new position is occupied by another walker
    if removed == False and temp_grid[i_n, j_n] == 2:
        # Prevent the walker from moving into occupied cell
        return temp_grid, i, j, removed

    return (temp_grid, i_n, j_n, removed)


def update_grid(grid):
    """
    Updates locations for all the random walkers
    and checks wheter the cluster is in their neighbourhood.
    """
    temp_grid = grid.copy()
    walkers_removed = 0

    for i in range(WIDTH):
        for j in range(WIDTH):
            if grid[i, j] == 2:  # Check if the cel contains a walker
                temp_grid, i_n, j_n, removed = new_indexes(temp_grid, grid, i, j)

                if removed:
                    temp_grid[i, j] = 0
                    walkers_removed += 1
                    continue
                cluster = check_neighbourhood(temp_grid, i_n, j_n)

                temp_grid[i, j] = 0
                # If there is a cluster in the new cel, it will become part of the
                # cluster, else the walker will move to the new spot.
                if cluster:
                    temp_grid[i_n, j_n] = 1
                else:
                    temp_grid[i_n, j_n] = 2

    # Add a new walker if any walker left the grid
    for _ in range(walkers_removed):
        temp_grid = add_walker(temp_grid)

    temp_grid = add_walker(temp_grid)
    grid[:] = temp_grid
    return grid


def start_simulation(seed, steps, width):
    """ Starts simulation with given parameter values. """
    global WIDTH
    WIDTH = width
    np.random.seed(seed)
    grid = initialize_grid(seed)
    all_grids = [grid.copy()]
    # print(grid, "\n")

    for i in range(steps - 1):
        # print("step: ", i+1)
        grid = update_grid(grid)
        all_grids.append(grid.copy())
        # print(grid, "\n")
    
    return all_grids
