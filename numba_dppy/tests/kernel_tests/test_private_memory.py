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
import numba_dpex
import numpy as np
import pytest
from numba_dpex.tests._helper import filter_strings


@pytest.mark.parametrize("filter_str", filter_strings)
def test_private_memory(filter_str):
    @numba_dpex.kernel
    def private_memory_kernel(A):
        i = numba_dpex.get_global_id(0)
        prvt_mem = numba_dpex.private.array(shape=1, dtype=np.float32)
        prvt_mem[0] = i
        numba_dpex.barrier(numba_dpex.CLK_LOCAL_MEM_FENCE)  # local mem fence
        A[i] = prvt_mem[0] * 2

    N = 64
    arr = np.zeros(N).astype(np.float32)
    orig = np.arange(N).astype(np.float32)

    with numba_dpex.offload_to_sycl_device(filter_str):
        private_memory_kernel[N, N](arr)

    # The computation is correct?
    np.testing.assert_allclose(orig * 2, arr)
