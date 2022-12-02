# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.core.kernel_interface.dispatcher import (
    Dispatcher,
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
        filter_str (str): The device name coming from filter_strings in ._helper.py
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

    d = Dispatcher(
        data_parallel_sum,
        array_access_specifiers=get_ordered_arg_access_types(
            data_parallel_sum, None
        ),
    )
    d.delete_cache()

    N = 10
    for i in range(N):
        d(a, b, c, global_range=[100])
    actual = dpt.asnumpy(c)
    d.delete_cache()

    assert np.all(expected == actual) and (d.cache_hits == N - 1)


@pytest.mark.skip(reason="only applicable for a non-CFD scenario")
@pytest.mark.parametrize("filter_str", filter_strings)
def test_caching_kernel_using_same_queue(filter_str):
    """Test kernel caching when the same queue is used to submit a kernel
    multiple times.

    Args:
        filter_str: SYCL filter selector string
    """
    global_size = 10
    N = global_size

    def data_parallel_sum(a, b, c):
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    with dpctl.device_context(filter_str) as gpu_queue:
        func = dpex.kernel(data_parallel_sum, enable_cache=True)
        cached_kernel = func[global_size, dpex.DEFAULT_LOCAL_SIZE].specialize(
            func._get_argtypes(a, b, c), gpu_queue
        )

        for i in range(10):
            _kernel = func[global_size, dpex.DEFAULT_LOCAL_SIZE].specialize(
                func._get_argtypes(a, b, c), gpu_queue
            )
            assert _kernel == cached_kernel


@pytest.mark.skip(reason="only applicable for a non-CFD scenario")
@pytest.mark.parametrize("filter_str", filter_strings)
def test_caching_kernel_using_same_context(filter_str):
    """Test kernel caching for the scenario where different SYCL queues that
    share a SYCL context are used to submit a kernel.

    Args:
        filter_str: SYCL filter selector string
    """
    global_size = 10
    N = global_size

    def data_parallel_sum(a, b, c):
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    # Set the global queue to the default device so that the cached_kernel gets
    # created for that device
    dpctl.set_global_queue(filter_str)
    func = dpex.kernel(data_parallel_sum)
    default_queue = dpctl.get_current_queue()
    cached_kernel = func[global_size, dpex.DEFAULT_LOCAL_SIZE].specialize(
        func._get_argtypes(a, b, c), default_queue
    )
    for i in range(0, 10):
        # Each iteration create a fresh queue that will share the same context
        with dpctl.device_context(filter_str) as gpu_queue:
            _kernel = func[global_size, dpex.DEFAULT_LOCAL_SIZE].specialize(
                func._get_argtypes(a, b, c), gpu_queue
            )
            assert _kernel == cached_kernel
