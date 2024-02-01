# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from numpy.testing import assert_equal

from numba_dpex import (
    DEFAULT_LOCAL_SIZE,
    NdRange,
    Range,
    atomic,
    get_global_id,
    get_global_size,
    get_group_id,
    get_local_id,
    get_local_size,
    local,
    mem_fence,
    private,
)
from numba_dpex.experimental.kernel_iface.simulator import kernel


def test_simple1():
    def func(a, b, c):
        i = get_global_id(0)
        j = get_global_id(1)
        k = get_global_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    sim_func = kernel(func)

    a = np.array([[[1, 2, 3], [4, 5, 6]]], np.float32)
    b = np.array([[[7, 8, 9], [10, 11, 12]]], np.float32)

    sim_res = np.zeros(a.shape, a.dtype)
    sim_func[a.shape, DEFAULT_LOCAL_SIZE](a, b, sim_res)

    expected_res = a + b

    assert_equal(sim_res, expected_res)
