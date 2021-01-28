import numba
import numpy as np
import dpctl


@numba.njit
def rand():
    return np.random.rand(3, 2)


@numba.njit
def random_sample(size):
    return np.random.random_sample(size)


@numba.njit
def random_exponential(scale, size):
    return np.random.exponential(scale, size)


@numba.njit
def random_normal(loc, scale, size):
    return np.random.normal(loc, scale, size)


size = 9
scale = 3.

with dpctl.device_context("opencl:gpu"):
    result = rand()
    # Random values in a given shape (3, 2)
    print(result)

    result = random_sample(size)
    # Array of shape (9,) with random floats in the half-open interval [0.0, 1.0)
    print(result)

    result = random_exponential(scale, size)
    # Array of shape (9,) with samples from an exponential distribution
    print(result)

    result = random_normal(0.0, 0.1, size)
    # Array of shape (9,) with samples from a normal distribution
    print(result)
