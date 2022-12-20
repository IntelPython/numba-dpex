# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import platform

import dpctl
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import filter_strings


@pytest.mark.parametrize("filter_str", filter_strings)
def test_proper_lowering(filter_str):
    # This will trigger eager compilation
    @dpex.kernel("void(float32[::1])")
    def twice(A):
        i = dpex.get_global_id(0)
        d = A[i]
        dpex.barrier(dpex.LOCAL_MEM_FENCE)  # local mem fence
        A[i] = d * 2

    N = 256
    arr = np.random.random(N).astype(np.float32)
    orig = arr.copy()

    with dpctl.device_context(filter_str):
        twice[N, N // 2](arr)

    # The computation is correct?
    np.testing.assert_allclose(orig * 2, arr)


@pytest.mark.parametrize("filter_str", filter_strings)
def test_no_arg_barrier_support(filter_str):
    @dpex.kernel("void(float32[::1])")
    def twice(A):
        i = dpex.get_global_id(0)
        d = A[i]
        # no argument defaults to global mem fence
        dpex.barrier()
        A[i] = d * 2

    N = 256
    arr = np.random.random(N).astype(np.float32)
    orig = arr.copy()

    with dpctl.device_context(filter_str):
        twice[N, dpex.DEFAULT_LOCAL_SIZE](arr)

    # The computation is correct?
    np.testing.assert_allclose(orig * 2, arr)


@pytest.mark.parametrize("filter_str", filter_strings)
def test_local_memory(filter_str):
    blocksize = 10

    @dpex.kernel("void(float32[::1])")
    def reverse_array(A):
        lm = dpex.local.array(shape=10, dtype=np.float32)
        i = dpex.get_global_id(0)

        # preload
        lm[i] = A[i]
        # barrier local or global will both work as we only have one work group
        dpex.barrier(dpex.LOCAL_MEM_FENCE)  # local mem fence
        # write
        A[i] += lm[blocksize - 1 - i]

    arr = np.arange(blocksize).astype(np.float32)
    orig = arr.copy()

    with dpctl.device_context(filter_str):
        reverse_array[blocksize, blocksize](arr)

    expected = orig[::-1] + orig
    np.testing.assert_allclose(expected, arr)
