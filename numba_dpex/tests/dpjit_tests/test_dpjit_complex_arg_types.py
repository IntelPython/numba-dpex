# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numba as nb
import numpy
import pytest

import numba_dpex as dpex

N = 1024


@dpex.dpjit
def prange_arg(a, b, c):
    for i in nb.prange(a.shape[0]):
        b[i] = a[i] * c


@dpex.dpjit
def prange_array(a, b, c):
    for i in nb.prange(a.shape[0]):
        b[i] = a[i] * c[i]


list_of_dtypes = [
    dpnp.complex64,
    dpnp.complex128,
]

list_of_usm_types = ["shared", "device", "host"]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = dpnp.ones(N, dtype=request.param)
    c = dpnp.zeros(N, dtype=request.param)
    b = dpnp.empty_like(a)
    return a, b, c


def test_dpjit_scalar_arg_types(input_arrays):
    """Tests passing float and complex type dpnp arrays to a dpjit prange function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    s = 2
    a, b, _ = input_arrays

    prange_arg(a, b, s)

    nb = dpnp.asnumpy(b)
    nexpected = numpy.full_like(nb, fill_value=2)

    assert numpy.allclose(nb, nexpected)


def test_dpjit_arg_complex_scalar(input_arrays):
    """Tests passing complex type scalar and dpnp arrays to a dpjit prange function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    s = 2 + 1j
    a, b, _ = input_arrays

    prange_arg(a, b, s)

    nb = dpnp.asnumpy(b)
    nexpected = numpy.full_like(nb, fill_value=2 + 1j)

    assert numpy.allclose(nb, nexpected)


def test_dpjit_arg_complex_array(input_arrays):
    """Tests passing complex type dpnp arrays to a dpjit prange function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """

    a, b, c = input_arrays

    prange_array(a, b, c)

    nb = dpnp.asnumpy(b)
    nexpected = numpy.full_like(nb, fill_value=0 + 0j)

    assert numpy.allclose(nb, nexpected)
