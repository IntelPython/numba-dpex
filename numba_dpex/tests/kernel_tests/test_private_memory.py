# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import platform

import dpctl
import numpy as np
import pytest

import numba_dpex
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
