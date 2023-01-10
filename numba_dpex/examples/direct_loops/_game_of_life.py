# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0
from time import time

import dpnp as np

from numba_dpex import njit, prange

N_UPDATES = 2000  # Perform N_UPDATES updates of the game

GRID_W, GRID_H = (
    1200,
    1200,
)  # 2D grid size. Each cell is in either ON (alive) or OFF (dead) state

PROB_ON = 0.2  # Initial probability that the cell is ON (i.e. alive)

# Conway's Game of Life rules
rules = np.array(
    [
        # 0  1  2  3  4  5  6  7  8   # Number of alive cell neighbors
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Rule for dead cells
        [0, 0, 1, 1, 0, 0, 0, 0, 0],  # Rule for alive cells
    ]
)


# Initial grid setting
def init_grid(w, h, p):
    return np.random.choice((0, 1), w * h, p=(1.0 - p, p)).reshape(
        h, w
    )  # Cells follow Bernoulli distribution


# Update grid cells according to Conway's Game of Life rules
@njit(["int32[:,:](int32[:,:])", "int64[:,:](int64[:,:])"], parallel=True)
def grid_update(grid):
    m, n = grid.shape
    grid_out = np.empty_like(grid)
    grid_padded = np.empty((m + 2, n + 2), dtype=grid.dtype)
    grid_padded[
        1:-1, 1:-1
    ] = grid  # Copy input grid into the center of padded one
    grid_padded[0, 1:-1] = grid[-1]  # Top row of padded grid
    grid_padded[-1, 1:-1] = grid[0]  # Bottom row of padded grid
    grid_padded[1:-1, 0] = grid[:, -1]  # Last column of padded grid
    grid_padded[1:-1, -1] = grid[:, 0]  # First column of padded grid
    grid_padded[0, 0] = grid[-1, -1]  # Bottom right cell
    grid_padded[-1, -1] = grid[0, 0]  # Top left cell
    grid_padded[0, -1] = grid[-1, 0]  # Bottom left cell
    grid_padded[-1, 0] = grid[0, -1]  # Top right cell

    for i in prange(m):  # Parallelize the outermost loop
        for j in range(n):
            v_self = grid[i, j]
            neighbor_population = (
                grid_padded[i : i + 3, j : j + 3].sum() - v_self
            )
            grid_out[i, j] = rules[v_self, neighbor_population]
    return grid_out


def main():
    grid = init_grid(GRID_W, GRID_H, PROB_ON)

    k = 0
    t1 = time()
    for k in range(N_UPDATES):
        grid_update(grid)
    t2 = time()
    print("Average FPS =", k / (t2 - t1))


if __name__ == "__main__":
    main()
