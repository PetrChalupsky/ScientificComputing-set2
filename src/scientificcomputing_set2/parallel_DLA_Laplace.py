import numpy as np
from numba import njit, cuda, prange
from scientificcomputing_set2.DLA_Laplace import determine_spread, initialize_grid


@cuda.jit
def update_red_cells(width, d_grid, d_new_grid, d_objects, omega):

    i, j = cuda.grid(2)
    if i > 0 and i < width - 1 and j > 0 and j < width - 1:

        if (i + j) % 2 == 0:
            if d_objects is not None and d_objects[i, j] == 1:
                d_new_grid[i, j] = 0
            else:
                d_new_grid[i, j] = (
                    0.25
                    * omega
                    * (
                        d_grid[(i + 1) % (width), j]
                        + d_grid[(i - 1) % (width), j]
                        + d_grid[i, (j - 1) % (width)]
                        + d_grid[i, (j + 1) % (width)]
                    )
                    + (1 - omega) * d_grid[i, j]
                )


@cuda.jit
def update_black_cells(width, d_grid, d_new_grid, d_objects, omega):

    i, j = cuda.grid(2)
    if i > 0 and i < width - 1 and j > 0 and j < width - 1:

        if (i + j) % 2 == 1:
            if d_objects is not None and d_objects[i, j] == 1:
                d_new_grid[i, j] = 0
            else:
                d_new_grid[i, j] = (
                    0.25
                    * omega
                    * (
                        d_grid[(i + 1) % (width), j]
                        + d_grid[(i - 1) % (width), j]
                        + d_grid[i, (j - 1) % (width)]
                        + d_grid[i, (j + 1) % (width)]
                    )
                    + (1 - omega) * d_grid[i, j]
                )


def sor_with_objects_cuda(width, eps, omega, objects, diffusion_grid):
    """
    GPU implementation of red and black SOR. Given the input makes an initial grid and updates this
    for a given time. Returns final grid.
    """

    # Initialize the grid
    if diffusion_grid is not None:
        new_grid = diffusion_grid
    else:
        new_grid = initialize_grid(width, False)

    # Copy the grid and objects to GPU
    d_new_grid = cuda.to_device(new_grid)

    d_objects = None
    if objects is not None:
        d_objects = cuda.to_device(objects)

    # Slice up the grid
    threads_per_block = (16, 16)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Update grid while difference larger than epsilon
    delta = 100
    delta_list = []
    k = 0

    while delta >= eps and k < 10000:
        d_old_grid = cuda.to_device(d_new_grid.copy_to_host())

        update_black_cells[blocks_per_grid, threads_per_block](
            width, d_new_grid, d_new_grid, d_objects, omega
        )

        cuda.synchronize()

        temp_grid = d_new_grid.copy_to_host()
        temp_grid[0, :] = 0
        temp_grid[width - 1, :] = 1
        d_new_grid = cuda.to_device(temp_grid)

        update_red_cells[blocks_per_grid, threads_per_block](
            width, d_new_grid, d_new_grid, d_objects, omega
        )
        new_grid = d_new_grid.copy_to_host()
        new_grid[0, :] = 0
        new_grid[width - 1, :] = 1
        d_new_grid = cuda.to_device(new_grid)

        old_grid = d_old_grid.copy_to_host()

        delta = np.max(np.abs(new_grid - old_grid))
        delta_list.append(delta)
        k = k + 1

    return d_new_grid.copy_to_host()


@njit(parallel=True)
def update_black_cells_parallel_cpu(width, grid, new_grid, objects, omega):
    for i in prange(1, width - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 1:

                if objects is not None and objects[i, j] == 1:
                    new_grid[i, j] = 0
                else:
                    new_grid[i, j] = (
                        0.25
                        * omega
                        * (
                            grid[(i + 1), j]
                            + grid[(i - 1), j]
                            + grid[i, (j - 1)]
                            + grid[i, (j + 1)]
                        )
                        + (1 - omega) * grid[i, j]
                    )
    return new_grid


@njit(parallel=True)
def update_red_cells_parallel_cpu(width, grid, new_grid, objects, omega):
    for i in prange(1, width - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 0:

                if objects is not None and objects[i, j] == 1:
                    new_grid[i, j] = 0
                else:
                    new_grid[i, j] = (
                        0.25
                        * omega
                        * (
                            grid[(i + 1), j]
                            + grid[(i - 1), j]
                            + grid[i, (j - 1)]
                            + grid[i, (j + 1)]
                        )
                        + (1 - omega) * grid[i, j]
                    )
    return new_grid


def sor_with_objects_parallel_cpu(width, eps, omega, objects, diffusion_grid):
    """
    SOR implementation using Numba for CPU parallelization.
    """
    # Initialize the grid
    if diffusion_grid is not None:
        new_grid = diffusion_grid.copy()
    else:
        new_grid = initialize_grid(width, False)

    # Iteration loop
    delta = 100
    delta_list = []
    k = 0

    while delta >= eps and k < 10000:
        grid = new_grid.copy()

        # Update black cells
        new_grid = update_black_cells_parallel_cpu(
            width, grid, new_grid.copy(), objects, omega
        )

        # Update red cells using the updated grid with new black values
        new_grid = update_red_cells_parallel_cpu(
            width, new_grid, new_grid.copy(), objects, omega
        )

        # Set boundary conditions
        new_grid[0, :] = 0
        new_grid[width - 1, :] = 1

        delta = np.max(np.abs(new_grid - grid))
        delta_list.append(delta)
        k += 1

    return new_grid


def run_cuda_DLA(width, eta, omega, cluster):
    eps = 0.00001
    diffusion_grid = None
    for _ in range(500):
        diffusion_grid = sor_with_objects_cuda(
            width, eps, omega, cluster, diffusion_grid
        )
        cluster = determine_spread(int(width), eta, diffusion_grid, cluster)

    cluster = np.array(cluster)

    return diffusion_grid, cluster


def run_parallel_cpu_DLA(width, eta, omega, cluster):
    eps = 0.00001
    diffusion_grid = None
    for _ in range(500):
        diffusion_grid = sor_with_objects_parallel_cpu(
            width, eps, omega, cluster, diffusion_grid
        )
        cluster = determine_spread(int(width), eta, diffusion_grid, cluster)

    cluster = np.array(cluster)

    return diffusion_grid, cluster
