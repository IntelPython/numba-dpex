# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numpy as np

import numba_dpex as ndpx


def make_write_values_kernel(n_rows):
    """Uppermost kernel to set 1s in a certain way. The uppermost kernel
    function invokes two levels of inner functions to set 1s in an empty matrix
    in a certain way.

    Args:
        n_rows (int): Number of rows to iterate.

    Returns:
        numba_dpex.core.kernel_interface.dispatcher.JitKernel:
            A KernelDispatcher object that encapsulates a @kernel decorated
            numba_dpex compiled kernel object.
    """
    write_values = make_write_values_kernel_func()

    @ndpx.kernel
    def write_values_kernel(array_in):
        for row_idx in range(n_rows):
            is_even = (row_idx % 2) == 0
            write_values(array_in, row_idx, is_even)

    return write_values_kernel


def make_write_values_kernel_func():
    """An upper function to set 1 or 3 ones. A function to set
    one or three 1s. If the row index is even it will set three 1s,
    otherwise one 1. It uses the inner function to do this.

    Returns:
        numba_dpex.core.kernel_interface.func.DpexFunctionTemplate:
            A DpexFunctionTemplate that encapsulates a @func decorated
            numba_dpex compiled function object.
    """
    write_when_odd = make_write_values_kernel_func_inner(1)
    write_when_even = make_write_values_kernel_func_inner(3)

    @ndpx.device_func
    def write_values(array_in, row_idx, is_even):
        if is_even:
            write_when_even(array_in, row_idx)
        else:
            write_when_odd(array_in, row_idx)

    return write_values


def make_write_values_kernel_func_inner(n_cols):
    """Inner function to set 1s. An inner function to set 1s in
    n_cols number of columns.

    Args:
        n_cols (int): Number of columns to be set to 1.

    Returns:
        numba_dpex.core.kernel_interface.func.DpexFunctionTemplate:
            A DpexFunctionTemplate that encapsulates a @func decorated
            numba_dpex compiled function object.
    """

    @ndpx.device_func
    def write_values_inner(array_in, row_idx):
        for idx in range(n_cols):
            array_in[row_idx, idx] = 1

    return write_values_inner


def test_qualname_basic():
    """A basic test function to test
    qualified name disambiguation.
    """
    ans = np.zeros((10, 10), dtype=np.int64)
    for i in range(ans.shape[0]):
        if i % 2 == 0:
            ans[i, 0:3] = 1
        else:
            ans[i, 0] = 1

    a = dpnp.zeros((10, 10), dtype=dpnp.int64)

    kernel = make_write_values_kernel(10)
    ndpx.call_kernel(kernel, ndpx.NdRange((1,), (1,)), a)

    assert np.array_equal(dpnp.asnumpy(a), ans)
