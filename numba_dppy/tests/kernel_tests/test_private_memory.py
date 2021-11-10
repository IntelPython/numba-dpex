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

import numba_dppy
from numba_dppy.tests._helper import skip_test

list_of_filter_strs = [
    "opencl:gpu:0",
    "level_zero:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


def test_private_memory(filter_str):
    if skip_test(filter_str):
        pytest.skip()

    @numba_dppy.kernel
    def private_memory_kernel(A):
        i = numba_dppy.get_global_id(0)
        prvt_mem = numba_dppy.private.array(shape=1, dtype=np.float32)
        prvt_mem[0] = i
        A[i] = prvt_mem[0] * 2

    N = 64
    arr = np.arange(N).astype(np.float32)
    orig = arr.copy()

    with numba_dppy.offload_to_sycl_device(filter_str):
        private_memory_kernel[N, N](arr)

    # The computation is correct?
    np.testing.assert_allclose(orig * 2, arr)
