# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest

import numba_dpex
from numba_dpex.tests._helper import filter_strings


@numba_dpex.kernel
def sum(a, b, c):
    i = numba_dpex.get_global_id(0)
    c[i] = a[i] + b[i]


@pytest.mark.parametrize("filter_str", filter_strings)
def test_strided_array_kernel(filter_str):
    global_size = 606
    a = np.arange(global_size * 2, dtype="i4")[::2]
    b = np.arange(global_size, dtype="i4")[::-1]
    got = np.zeros(global_size, dtype="i4")

    expected = a + b

    with dpctl.device_context(filter_str):
        sum[global_size, numba_dpex.DEFAULT_LOCAL_SIZE](a, b, got)

    assert np.array_equal(expected, got)
