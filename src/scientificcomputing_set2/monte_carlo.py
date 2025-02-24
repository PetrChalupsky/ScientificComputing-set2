"""
Course: Scientific computing
Names: Lisa Pijpers, Petr Chalupský and Tika van Bennekum
Student IDs: 15746704, 15719227 and 13392425

File description:
    Monte Carlo implementation of the DLA, using random walkers.
"""

import numpy as np

def initialize_grid(seed, width):
    """
    Initialize grid given a width as parameter. It assumes a square grid.
    A randomly selected cell on the bottom row is the starting cluster.
    """
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
    np.random.seed(seed)
    random_col = np.random.randint(0, len(grid))
    if grid[0, random_col] != 1:
        grid[0, random_col] = 2

    return grid

def update_grid(seed, grid):
    """
    Updates locations for all the random walkers
    and checks wheter the cluster is in their neighbourhood.
    """
    np.random.random(seed)
    copy_grid = grid.copy()

    for i in grid:
        for j in grid:
            # randomly choose wheter i + 1, i - 1, j + 1 or j - 1
            if grid[i, j] == 2:  # Check if the cel containts a walker
                direction = np.random.choice(["up", "down", "left", "right"])

                # Compute new position
                i_n, j_n = i, j
                # Random walker goes up or down, if it goes out of bounds,
                # it is removed and a new random walker is created.
                if direction == "up":
                    if i > 0:
                        i_n -= 1
                    else:
                        grid[i, j] = 0
                        grid = add_walker(grid)
                if direction == "down":
                    if i < 0:
                        i_n += 1
                    else:
                        grid[i, j] = 0
                        grid = add_walker(grid)
                # Random walker goes left or right, if it goes out of bounds,
                # it reapears on the other side of the grid.
                elif direction == "left":
                    if j > 0:
                        j_n -= 1
                    else:
                        j_n = len(grid) - 1
                elif direction == "right":
                    if j < len(grid) - 1:
                        j_n += 1
                    else:
                        j_n = 0

                # Move the walker to the new position
                copy_grid[i, j] = 0  # Remove from old position
                copy_grid[i_n, j_n] = 1  # Place at new position



