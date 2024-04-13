# SPDX-FileCopyrightText: 2022 - 2023 Julien Jerphanion <git@jjerphan.xyz>
# SPDX-FileCopyrightText: 2022 - 2023 Olivier Grisel <olivier.grisel@ensta.org>
# SPDX-FileCopyrightText: 2022 - 2023 Franck Charras <29153872+fcharras@users.noreply.github.com>
# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This code is largely inspired from the numba.cuda.random module and the
# numba/cuda/random.py where it's defined (v<0.57), and by the implementation of the
# same algorithm in the package `randomgen`.

# numba.cuda.random: https://github.com/numba/numba/blob/0.56.3/numba/cuda/random.py
# randomgen: https://github.com/bashtage/randomgen/blob/v1.26.0/randomgen/xoroshiro128.pyx  # noqa

# NB1: we implement xoroshiro128++ rather than just xoroshiro128+, which is preferred.
# Reference resource about PRNG: https://prng.di.unimi.it/

# NB2: original numba.cuda.random code also includes functions for generating normally
# distributed floats but we don't include it here as long as it's not needed.
"""

import random
import warnings
from functools import lru_cache

import dpctl
import dpctl.tensor as dpt
import numpy as np
from numba import float32, float64, int64, uint32, uint64

import numba_dpex as dpex


def _get_sequential_processing_device(device: dpctl.SyclDevice):
    """Returns a device most fitted for sequential processing.

    Selects a cpu rather than a gpu for sequential processing. If such a cpu
    device is not found, returns the input device instead.

    Also returns a boolean that informs on wether the returned device is
    different than the input device.
    """
    if device.has_aspect_cpu:
        return device, False

    try:
        return dpctl.SyclDevice("cpu"), True
    except dpctl.SyclDeviceCreationError:
        warnings.warn(
            "No CPU found, falling back to GPU for sequential instructions."
        )
        return device, False


zero_idx = int64(0)
one_idx = int64(1)


def get_random_raw(states):
    """Returns a single pseudo-random `uint64` integer value.

    Similar to numpy.random.BitGenerator.random_raw(size=1).

    Note, this always uses and updates state states[0].
    """
    result = dpt.empty((1,), dtype=np.uint64, device=states.device)
    dpex.call_kernel(make_random_raw_kernel(), dpex.Range(1), states, result)
    return result


@lru_cache
def make_random_raw_kernel():
    """Returns a single pseudo-random `uint64` integer value.
    Similar to numpy.random.BitGenerator.random_raw(size=1).

    Note, this always uses and updates state states[0].
    """

    @dpex.kernel
    def _get_random_raw_kernel(
        item, states, result
    ):  # pylint: disable=unused-argument
        result[zero_idx] = _xoroshiro128pp_next(states, zero_idx)

    return _get_random_raw_kernel


def make_rand_uniform_kernel_func(dtype):
    """Instantiate a kernel function that returns a random float in [0, 1)

    This factory returns a kernel function specialized to either generate
    float32 or float64 values. Care has been taken so that the float32
    variant can be compiled to target a device that does not support the
    float64 aspect.

    The returned kernel function takes two arguments:

    - states : a state array. See create_xoroshiro128pp_states for details.

    - state_idx : the index of the RNG state to use to generate the next
      random float.

    """
    if not hasattr(dtype, "name"):
        raise ValueError(
            "dtype is expected to have an attribute 'name', like np.dtype "
            "or numba types."
        )

    if dtype.name == "float64":
        convert_rshift = uint32(11)
        convert_const = float64(uint64(1) << uint32(53))
        convert_const_one = float64(1)

        @dpex.device_func
        def uint64_to_unit_float(x):
            """Convert uint64 to float64 value in the range [0.0, 1.0)"""
            return float64(x >> convert_rshift) * (
                convert_const_one / convert_const
            )

    elif dtype.name == "float32":
        convert_rshift = uint32(40)
        convert_const = float32(uint32(1) << uint32(24))
        convert_const_one = float32(1)

        @dpex.device_func
        def uint64_to_unit_float(x):
            """Convert uint64 to float32 value in the range [0.0, 1.0)

            NB: this is different than original numba.cuda.random code. Instead
            of generating a float64 random number before casting it to float32,
            a float32 number is generated from uint64 without intermediate
            float64. This change enables compatibility with devices that do not
            support float64 numbers. However is seems to be exactly equivalent
            e.g it passes the float precision test in sklearn.
            """
            return float32(x >> convert_rshift) * (
                convert_const_one / convert_const
            )

    else:
        raise ValueError(
            "Expected dtype.name in {float32, float64} but got "
            f"dtype.name == {dtype.name}"
        )

    @dpex.device_func
    def xoroshiro128pp_uniform(states, state_idx):
        """Return one random float in [0, 1)

        Calling this function advances the states[state_idx] by a single RNG
        step and leaves the other states unchanged.
        """
        return uint64_to_unit_float(_xoroshiro128pp_next(states, state_idx))

    return xoroshiro128pp_uniform


def create_xoroshiro128pp_states(
    n_states, subsequence_start=0, seed=None, device=None
):
    """Returns a new device array initialized for n random number generators.

    This initializes the RNG states so that states in the array correspond to
    subsequences separated by 2**64 steps from each other in the main sequence.
    Therefore, as long as no thread requests more than 2**64 random numbers, all
    the RNG states produced by this function are guaranteed to be independent.

    Parameters
    ----------
    n_states : int
        Number of RNG states to create. Each RNG state is meant to be used by a
        distinct thread in the xoroshiro128pp RNG. Therefore n_states controls
        the amount of parallelism when using the RNG to generate a large enough
        sequence of pseudo-random values. Subsequent states are initialized
        2**64 RNG steps away from one another.

    subsequence_start : int
        Advance the first RNG state by a multiple of 2**64 steps after the
        state induced by the seed.  The subsequent RNG states controlled by
        `n_states` are each initialized 2**64 steps further from their
        predecessor in the states array.

    seed : int or None
        Starting seed for the list of generators.

    device : str or None (default)
        A SYCL device or if None, takes the default sycl device.
    """
    if seed is None:
        seed = uint64(random.randint(0, np.iinfo(np.int64).max - 1))

    if hasattr(seed, "randint"):
        seed = uint64(seed.randint(0, np.iinfo(np.int64).max - 1))

    init_xoroshiro128pp_states_kernel = _make_init_xoroshiro128pp_states_kernel(
        n_states, subsequence_start
    )

    # Initialization is purely sequential so it will be faster on CPU, if a
    # cpu device is available make sure to use it.
    if device is None:
        device = dpctl.SyclDevice()

    (
        sequential_processing_device,
        sequential_processing_on_different_device,
    ) = _get_sequential_processing_device(device)

    states = dpt.empty(
        (n_states, 2), dtype=np.uint64, device=sequential_processing_device
    )

    seed = dpt.asarray(
        [seed], dtype=np.uint64, device=sequential_processing_device
    )

    dpex.call_kernel(
        init_xoroshiro128pp_states_kernel, dpex.Range(1), states, seed
    )

    if sequential_processing_on_different_device:
        return states.to_device(device)

    return states


@lru_cache
def _make_init_xoroshiro128pp_states_kernel(
    n_states, subsequence_start
):  # pylint: disable=too-many-locals
    n_states = int64(n_states)

    splitmix64_const_1 = uint64(0x9E3779B97F4A7C15)
    splitmix64_const_2 = uint64(0xBF58476D1CE4E5B9)
    splitmix64_const_3 = uint64(0x94D049BB133111EB)
    splitmix64_rshift_1 = uint32(30)
    splitmix64_rshift_2 = uint32(27)
    splitmix64_rshift_3 = uint32(31)

    @dpex.device_func
    def _splitmix64_next(state):
        new_state = z = state + splitmix64_const_1
        z = (z ^ (z >> splitmix64_rshift_1)) * splitmix64_const_2
        z = (z ^ (z >> splitmix64_rshift_2)) * splitmix64_const_3
        return new_state, z ^ (z >> splitmix64_rshift_3)

    jump_const_1 = uint64(0x2BD7A6A6E99C2DDC)
    jump_const_2 = uint64(0x0992CCAF6A6FCA05)
    jump_const_3 = uint64(1)
    jump_init = uint64(0)
    long_2 = int64(2)
    long_64 = int64(64)

    @dpex.device_func
    def _xoroshiro128pp_jump(states, state_idx):
        """Advance the RNG in ``states[state_idx]`` by 2**64 steps."""
        s0 = jump_init
        s1 = jump_init

        for i in range(long_2):
            if i == zero_idx:
                jump_const = jump_const_1
            else:
                jump_const = jump_const_2
            for b in range(long_64):
                if jump_const & jump_const_3 << uint32(b):
                    s0 ^= states[state_idx, zero_idx]
                    s1 ^= states[state_idx, one_idx]
                _xoroshiro128pp_next(states, state_idx)

        states[state_idx, zero_idx] = s0
        states[state_idx, one_idx] = s1

    init_const_1 = np.uint64(0)

    @dpex.kernel
    def init_xoroshiro128pp_states(
        item, states, seed
    ):  # pylint: disable=unused-argument
        """
        Use SplitMix64 to generate an xoroshiro128p state from a uint64 seed.

        This ensures that manually set small seeds don't result in a predictable
        initial sequence from the random number generator.
        """
        if n_states < one_idx:
            return

        splitmix64_state = init_const_1 ^ seed[zero_idx]
        splitmix64_state, states[zero_idx, zero_idx] = _splitmix64_next(
            splitmix64_state
        )
        _, states[zero_idx, one_idx] = _splitmix64_next(splitmix64_state)

        # advance to starting subsequence number
        for _ in range(subsequence_start):
            _xoroshiro128pp_jump(states, zero_idx)

        # populate the rest of the array
        for idx in range(one_idx, n_states):
            # take state of previous generator
            states[idx, zero_idx] = states[idx - one_idx, zero_idx]
            states[idx, one_idx] = states[idx - one_idx, one_idx]
            # and jump forward 2**64 steps
            _xoroshiro128pp_jump(states, idx)

    return init_xoroshiro128pp_states


_64_as_uint32 = uint32(64)


@dpex.device_func
def _rotl(x, k):
    """Left rotate x by k bits. x is expected to be a uint64 integer."""
    return (x << k) | (x >> (_64_as_uint32 - k))


next_rot_1 = uint32(17)
next_rot_2 = uint32(49)
next_rot_3 = uint32(28)
shift_1 = uint32(21)


@dpex.device_func
def _xoroshiro128pp_next(states, state_idx):
    """
    Returns the next random uint64 and advance the RNG in states[state_idx].
    """
    s0 = states[state_idx, zero_idx]
    s1 = states[state_idx, one_idx]
    result = _rotl(s0 + s1, next_rot_1) + s0

    s1 ^= s0
    states[state_idx, zero_idx] = _rotl(s0, next_rot_2) ^ s1 ^ (s1 << shift_1)
    states[state_idx, one_idx] = _rotl(s1, next_rot_3)

    return result
