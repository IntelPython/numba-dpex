# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

import dpnp as np

import numba_dpex as ndpx

BATCH_SIZE = 2048  # Draw random pairs (x, y) in batches of BATCH_SIZE elements
N_BATCHES = 1024  # Total number of batches
N = N_BATCHES * BATCH_SIZE  # Total number of random pairs (x, y)


# Array-style implementation
# This JIT function will be compiled once on its first invocation and then
# called many times to get an improved estimate for Pi
@ndpx.njit()
def monte_carlo_pi_batch():
    x = np.random.random(BATCH_SIZE)
    y = np.random.random(BATCH_SIZE)
    acc = np.count_nonzero(x * x + y * y <= 1.0)
    return 4.0 * acc / BATCH_SIZE


# Pi is estimated by generating 2d random points in (0,1) in small batches of BATCH_SIZE size.
# Each batch estimate is then averaged N_BATCHES times, and the final Pi estimate is returned
def monte_carlo_pi():
    s = 0.0
    for i in range(N_BATCHES):
        s += monte_carlo_pi_batch()
    return s / N_BATCHES


def main():
    print("Using device ...")
    pi = monte_carlo_pi()
    print(pi.device)
    print("Pi =", pi)
    print("Total", N, "points generated")
    print("Done ...")


if __name__ == "__main__":
    main()
