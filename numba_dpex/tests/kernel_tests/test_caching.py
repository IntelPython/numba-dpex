# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.core.kernel_interface.dispatcher import (
    JitKernel,
    get_ordered_arg_access_types,
)
from numba_dpex.tests._helper import filter_strings


@pytest.mark.parametrize("filter_str", filter_strings)
def test_caching_hit_counts(filter_str):
    """Tests the correct number of cache hits.
    If a Dispatcher is invoked 10 times and if the caching is enabled,
    then the total number of cache hits will be 9. Given the fact that
    the first time the kernel will be compiled and it will be loaded
    off the cache for the next time on.

    Args:
        filter_str (str): The device name coming from filter_strings in
        ._helper.py
    """

    def data_parallel_sum(x, y, z):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dpex.get_global_id(0)
        z[i] = x[i] + y[i]

    a = dpt.arange(0, 100, device=filter_str)
    b = dpt.arange(0, 100, device=filter_str)
    c = dpt.zeros_like(a, device=filter_str)

    expected = dpt.asnumpy(a) + dpt.asnumpy(b)

    d = JitKernel(
        data_parallel_sum,
        array_access_specifiers=get_ordered_arg_access_types(
            data_parallel_sum, None
        ),
    )

    d_launcher = d[100]

    N = 10
    for i in range(N):
        d_launcher(a, b, c)
    actual = dpt.asnumpy(c)

    assert np.array_equal(expected, actual) and (d_launcher.cache_hits == N - 1)
