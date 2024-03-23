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
def kernel_scalar(item, a, b, c):
    i = item.get_id(0)
    b[i] = a[i] * c


@dpex.kernel
def kernel_array(item, a, b, c):
    i = item.get_id(0)
    b[i] = a[i] * c[i]


list_of_dtypes = get_all_dtypes(
    no_bool=True, no_int=True, no_float=True, no_none=True
)

list_of_usm_types = ["shared", "device", "host"]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = dpnp.ones(N, dtype=request.param)
    c = dpnp.zeros(N, dtype=request.param)
    b = dpnp.empty_like(a)
    return a, b, c


def test_numeric_kernel_arg_complex_scalar(input_arrays):
    """Tests passing complex type scalar and dpnp arrays to a kernel function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    a, b, _ = input_arrays
    s = a.dtype.type(2 + 1j)

    dpex.call_kernel(kernel_scalar, dpex.Range(N), a, b, s)

    nb = dpnp.asnumpy(b)
    nexpected = numpy.full_like(nb, fill_value=2 + 1j)

    assert numpy.allclose(nb, nexpected)


def test_numeric_kernel_arg_complex_array(input_arrays):
    """Tests passing complex type dpnp arrays to a kernel function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """

    a, b, c = input_arrays

    dpex.call_kernel(kernel_array, dpex.Range(N), a, b, c)

    nb = dpnp.asnumpy(b)
    nexpected = numpy.full_like(nb, fill_value=0 + 0j)

    assert numpy.allclose(nb, nexpected)
