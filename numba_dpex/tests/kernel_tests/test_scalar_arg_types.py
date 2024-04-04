# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numpy
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import get_all_dtypes

N = 1024


@dpex.kernel
def scaling_kernel(item, a, b, c):
    i = item.get_id(0)
    b[i] = a[i] * c


@dpex.kernel
def kernel_with_bool_arg(item, a, b, test):
    i = item.get_id(0)
    if test:
        b[i] = a[i] + a[i]
    else:
        b[i] = a[i] - a[i]


list_of_dtypes = get_all_dtypes(no_bool=True, no_float16=True, no_none=True)

list_of_usm_types = ["shared", "device", "host"]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = dpnp.ones(N, dtype=request.param)
    b = dpnp.empty_like(a)
    return a, b


def test_numeric_kernel_arg_types1(input_arrays):
    """Tests passing float, int and complex type dpnp arrays to a kernel
    function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    a, b = input_arrays
    s = a.dtype.type(2)

    dpex.call_kernel(scaling_kernel, dpex.Range(N), a, b, s)

    nb = dpnp.asnumpy(b)
    nexpected = numpy.full_like(nb, fill_value=2)

    assert numpy.allclose(nb, nexpected)


def test_bool_kernel_arg_type(input_arrays):
    """Tests passing boolean arguments to a kernel function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    a, b = input_arrays

    dpex.call_kernel(kernel_with_bool_arg, dpex.Range(a.size), a, b, True)

    nb = dpnp.asnumpy(b)
    nexpected_true = numpy.full_like(nb, fill_value=2)

    assert numpy.allclose(nb, nexpected_true)

    dpex.call_kernel(kernel_with_bool_arg, dpex.Range(a.size), a, b, False)

    nb = dpnp.asnumpy(b)
    nexpected_false = numpy.zeros_like(nb)

    assert numpy.allclose(nb, nexpected_false)
