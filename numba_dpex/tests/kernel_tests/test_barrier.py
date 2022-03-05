# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform

import dpctl
import numpy as np
import pytest

import numba_dpex as dppy
from numba_dpex.tests._helper import filter_strings


@pytest.mark.parametrize("filter_str", filter_strings)
def test_proper_lowering(filter_str):
    # This will trigger eager compilation
    @dppy.kernel("void(float32[::1])")
    def twice(A):
        i = dppy.get_global_id(0)
        d = A[i]
        dppy.barrier(dppy.CLK_LOCAL_MEM_FENCE)  # local mem fence
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
    @dppy.kernel("void(float32[::1])")
    def twice(A):
        i = dppy.get_global_id(0)
        d = A[i]
        # no argument defaults to global mem fence
        dppy.barrier()
        A[i] = d * 2

    N = 256
    arr = np.random.random(N).astype(np.float32)
    orig = arr.copy()

    with dpctl.device_context(filter_str):
        twice[N, dppy.DEFAULT_LOCAL_SIZE](arr)

    # The computation is correct?
    np.testing.assert_allclose(orig * 2, arr)


@pytest.mark.parametrize("filter_str", filter_strings)
def test_local_memory(filter_str):
    blocksize = 10

    @dppy.kernel("void(float32[::1])")
    def reverse_array(A):
        lm = dppy.local.array(shape=10, dtype=np.float32)
        i = dppy.get_global_id(0)

        # preload
        lm[i] = A[i]
        # barrier local or global will both work as we only have one work group
        dppy.barrier(dppy.CLK_LOCAL_MEM_FENCE)  # local mem fence
        # write
        A[i] += lm[blocksize - 1 - i]

    arr = np.arange(blocksize).astype(np.float32)
    orig = arr.copy()

    with dpctl.device_context(filter_str):
        reverse_array[blocksize, blocksize](arr)

    expected = orig[::-1] + orig
    np.testing.assert_allclose(expected, arr)
