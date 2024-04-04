# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp

import numba_dpex as dpex
from numba_dpex import NdRange, Range, dpjit


@dpex.kernel(
    release_gil=False,
    no_compile=True,
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True,
)
def add(a, b, c):
    c[0] = b[0] + a[0]


@dpex.kernel(
    release_gil=False,
    no_compile=True,
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True,
)
def sq(a, b):
    a[0] = b[0] * b[0]


def test_call_kernel_from_cpython():
    """
    Tests if we can call a kernel function from CPython using the call_kernel
    dpjit function.
    """

    q = dpctl.SyclQueue()
    a = dpnp.ones(100, sycl_queue=q)
    b = dpnp.ones_like(a, sycl_queue=q)
    c = dpnp.zeros_like(a, sycl_queue=q)
    r = Range(100)
    ndr = NdRange(global_size=(100,), local_size=(1,))

    dpex.call_kernel(add, r, a, b, c)

    assert c[0] == b[0] + a[0]

    dpex.call_kernel(add, ndr, a, b, c)

    assert c[0] == b[0] + a[0]


def test_call_kernel_from_dpjit():
    """
    Tests if we can call a kernel function from a dpjit function using the
    call_kernel dpjit function.
    """

    @dpjit
    def range_kernel_caller(q, a, b, c):
        r = Range(100)
        dpex.call_kernel(add, r, a, b, c)
        return c

    @dpjit
    def ndrange_kernel_caller(q, a, b, c):
        gr = Range(100)
        lr = Range(1)
        ndr = NdRange(gr, lr)
        dpex.call_kernel(add, ndr, a, b, c)
        return c

    q = dpctl.SyclQueue()
    a = dpnp.ones(100, sycl_queue=q)
    b = dpnp.ones_like(a, sycl_queue=q)
    c = dpnp.zeros_like(a, sycl_queue=q)

    range_kernel_caller(q, a, b, c)

    assert c[0] == b[0] + a[0]

    ndrange_kernel_caller(q, a, b, c)

    assert c[0] == b[0] + a[0]


def test_call_multiple_kernels():
    """
    Tests if the call_kernel dpjit function supports calling different types of
    kernel with different number of arguments.
    """
    q = dpctl.SyclQueue()
    a = dpnp.ones(100, sycl_queue=q)
    b = dpnp.ones_like(a, sycl_queue=q)
    c = dpnp.zeros_like(a, sycl_queue=q)
    r = Range(100)

    dpex.call_kernel(add, r, a, b, c)

    assert c[0] == b[0] + a[0]

    dpex.call_kernel(sq, r, a, c)

    assert a[0] == c[0] * c[0]
