"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Monte Carlo implementation of the DLA, using random walkers.
    A cell has the value 0 when empty, 1 when containing the cluster and 2 when containing a walker.
"""

import numpy as np

global WIDTH
global SPACE


def initialize_grid(seed, width):
    """
    Initialize grid given a width as parameter. It assumes a square grid.
    A randomly selected cell on the bottom row is the starting cluster.
    """
    global WIDTH
    global SPACE
    WIDTH = width
    SPACE = True

    # Set empty grid
    grid = np.zeros((width, width))

    # Initialize the cluster
    np.random.seed(seed)
    random_col = np.random.randint(0, width)
    grid[-1, random_col] = 1

    return grid


def add_walker(seed, grid):
    """
    Adds a Random Walker to the grid on the top boundary.
    """
    global SPACE
    empty_cells = np.where(grid[0] == 0)[0]

    if empty_cells.size > 0:
        random_col = np.random.choice(empty_cells)
        grid[0, random_col] = 2
    else:
        SPACE = False
        raise ValueError("No new walkers can be added to the system!")

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


def new_indexes(temp_grid, i, j):
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

    if direction == "up":
        if i > 0:
            i_n -= 1
        else:
            temp_grid[i, j] = 0
            if SPACE:
                temp_grid = add_walker(temp_grid)
    if direction == "down":
        if i < WIDTH - 1:
            i_n += 1
        else:
            temp_grid[i, j] = 0
            if SPACE:
                temp_grid = add_walker(temp_grid)
    elif direction == "left":
        if j > 0:
            j_n -= 1
        else:
            j_n = WIDTH - 1
    elif direction == "right":
        if j < WIDTH - 1:
            j_n += 1
        else:
            j_n = 0

    # Check if the new position is occupied by another walker
    if temp_grid[i_n, j_n] == 2:
        # Prevent the walker from moving into occupied cell
        return temp_grid, i, j

    return (temp_grid, i_n, j_n)


def update_grid(seed, grid):
    """
    Updates locations for all the random walkers
    and checks wheter the cluster is in their neighbourhood.
    """
    np.random.seed(seed)
    temp_grid = grid.copy()

    for i in range(WIDTH):
        for j in range(WIDTH):
            if grid[i, j] == 2:  # Check if the cel contains a walker
                temp_grid, i_n, j_n = new_indexes(temp_grid, i, j)
                cluster = check_neighbourhood(temp_grid, i_n, j_n)

                temp_grid[i, j] = 0
                # If there is a cluster in the new cel, it will become part of the
                # cluster, else the walker will move to the new spot.
                if cluster:
                    temp_grid[i_n, j_n] = 1
                else:
                    temp_grid[i_n, j_n] = 2

    grid[:] = temp_grid
    return grid


def start_simulation(seed, steps, width):
    grid = initialize_grid(seed, width)

    for _ in range(steps):
        grid = update_grid(seed, grid)
