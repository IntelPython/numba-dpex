# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import dpnp
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.kernel_api import Item, Range


def get_order(a):
    """Get order of an array.

    Args:
        a (numpy.ndarray, dpnp.ndarray): Input array.

    Raises:
        Exception: _description_

    Returns:
        str: 'C' if c-contiguous, 'F' if f-contiguous or 'A' if aligned.
    """
    if a.flags.c_contiguous and not a.flags.f_contiguous:
        return "C"
    elif not a.flags.c_contiguous and a.flags.f_contiguous:
        return "F"
    elif a.flags.c_contiguous and a.flags.f_contiguous:
        return "A"
    else:
        raise Exception("Unknown order/layout")


@dpex.kernel
def change_values_1d(item: Item, x, v):
    """Assign values in a 1d dpnp.ndarray

    Args:
        x (dpnp.ndarray): Input array.
        v (int): Value to be assigned.
    """
    i = item.get_id(0)
    x[i] = v


def change_values_1d_func(a, p):
    """Assign values in a 1d numpy.ndarray

    Args:
        a (numpy.ndarray): Input array.
        p (int): Value to be assigned.
    """
    for i in range(a.shape[0]):
        a[i] = p


@dpex.kernel
def change_values_2d(item: Item, x, v):
    """Assign values in a 2d dpnp.ndarray

    Args:
        x (dpnp.ndarray): Input array.
        v (int): Value to be assigned.
    """
    i = item.get_id(0)
    j = item.get_id(1)
    x[i, j] = v


def change_values_2d_func(a, p):
    """Assign values in a 2d numpy.ndarray

    Args:
        a (numpy.ndarray): Input array.
        p (int): Value to be assigned.
    """
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] = p


@dpex.kernel
def change_values_3d(item: Item, x, v):
    """Assign values in a 3d dpnp.ndarray

    Args:
        x (dpnp.ndarray): Input array.
        v (int): Value to be assigned.
    """
    i = item.get_id(0)
    j = item.get_id(1)
    k = item.get_id(2)
    x[i, j, k] = v


def change_values_3d_func(a, p):
    """Assign values in a 3d numpy.ndarray

    Args:
        a (numpy.ndarray): Input array.
        p (int): Value to be assigned.
    """
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                a[i, j, k] = p


@pytest.mark.parametrize("s", [1, 2, 3, 4, 5, 6, 7])
def test_1d_strided_dpnp_array_in_kernel(s):
    """
    Tests if we can correctly handle a strided 1d dpnp array
    inside dpex kernel.
    """
    N = 256
    k = -3

    t = np.arange(0, N, dtype=dpnp.int64)
    u = dpnp.asarray(t)

    v = u[::s]
    dpex.call_kernel(change_values_1d, Range(v.shape[0]), v, k)

    x = t[::s]
    change_values_1d_func(x, k)

    # check the value of the array view
    assert np.all(dpnp.asnumpy(v) == x)
    # check the value of the original arrays
    assert np.all(dpnp.asnumpy(u) == t)


@pytest.mark.parametrize("s", [2, 3, 4, 5])
def test_multievel_1d_strided_dpnp_array_in_kernel(s):
    """
    Tests if we can correctly handle a multilevel strided 1d dpnp array
    inside dpex kernel.
    """
    N = 256
    k = -3

    t = dpnp.arange(0, N, dtype=dpnp.int64)
    u = dpnp.asarray(t)

    v, x = u, t
    while v.shape[0] > 1:
        v = v[::s]
        dpex.call_kernel(change_values_1d, Range(v.shape[0]), v, k)

        x = x[::s]
        change_values_1d_func(x, k)

        # check the value of the array view
        assert np.all(dpnp.asnumpy(v) == x)
        # check the value of the original arrays
        assert np.all(dpnp.asnumpy(u) == t)


@pytest.mark.parametrize("s1", [2, 4, 6, 8])
@pytest.mark.parametrize("s2", [1, 3, 5, 7])
@pytest.mark.parametrize("order", ["C", "F"])
def test_2d_strided_dpnp_array_in_kernel(s1, s2, order):
    """
    Tests if we can correctly handle a strided 2d dpnp array
    inside dpex kernel.
    """
    M, N = 13, 31
    k = -3

    t = np.arange(0, M * N, dtype=np.int64).reshape(M, N, order=order)
    u = dpnp.asarray(t)

    # check order, sanity check
    assert get_order(u) == order

    v = u[::s1, ::s2]
    dpex.call_kernel(change_values_2d, Range(*v.shape), v, k)

    x = t[::s1, ::s2]
    change_values_2d_func(x, k)

    # check the value of the array view
    assert np.all(dpnp.asnumpy(v) == x)
    # check the value of the original arrays
    assert np.all(dpnp.asnumpy(u) == t)


@pytest.mark.parametrize("s1", [2, 4, 6, 8])
@pytest.mark.parametrize("s2", [3, 5, 7, 9])
@pytest.mark.parametrize("order", ["C", "F"])
def test_multilevel_2d_strided_dpnp_array_in_kernel(s1, s2, order):
    """
    Tests if we can correctly handle a multilevel strided 2d dpnp array
    inside dpex kernel.
    """
    M, N = 13, 31
    k = -3

    t = np.arange(0, M * N, dtype=np.int64).reshape(M, N, order=order)
    u = dpnp.asarray(t)

    # check order, sanity check
    assert get_order(u) == order

    v, x = u, t
    while v.shape[0] > 1 and v.shape[1] > 1:
        v = v[::s1, ::s2]
        dpex.call_kernel(change_values_2d, Range(*v.shape), v, k)

        x = x[::s1, ::s2]
        change_values_2d_func(x, k)

        # check the value of the array view
        assert np.all(dpnp.asnumpy(v) == x)
        # check the value of the original arrays
        assert np.all(dpnp.asnumpy(u) == t)


@pytest.mark.parametrize("s1", [1, 2, 3])
@pytest.mark.parametrize("s2", [2, 3, 4])
@pytest.mark.parametrize("s3", [3, 4, 5])
@pytest.mark.parametrize("order", ["C", "F"])
def test_3d_strided_dpnp_array_in_kernel(s1, s2, s3, order):
    """
    Tests if we can correctly handle a strided 3d dpnp array
    inside dpex kernel.
    """
    M, N, K = 13, 31, 11
    k = -3

    t = np.arange(0, M * N * K, dtype=np.int64).reshape((M, N, K), order=order)
    u = dpnp.asarray(t)

    # check order, sanity check
    assert get_order(u) == order

    v = u[::s1, ::s2, ::s3]
    dpex.call_kernel(change_values_3d, Range(*v.shape), v, k)

    x = t[::s1, ::s2, ::s3]
    change_values_3d_func(x, k)

    # check the value of the array view
    assert np.all(dpnp.asnumpy(v) == x)
    # check the value of the original arrays
    assert np.all(dpnp.asnumpy(u) == t)


@pytest.mark.parametrize("s1", [2, 3, 4])
@pytest.mark.parametrize("s2", [3, 4, 5])
@pytest.mark.parametrize("s3", [4, 5, 6])
@pytest.mark.parametrize("order", ["C", "F"])
def test_multilevel_3d_strided_dpnp_array_in_kernel(s1, s2, s3, order):
    """
    Tests if we can correctly handle a multilevel strided 3d dpnp array
    inside dpex kernel.
    """
    M, N, K = 13, 31, 11
    k = -3

    t = np.arange(0, M * N * K, dtype=np.int64).reshape((M, N, K), order=order)
    u = dpnp.asarray(t)

    # check order, sanity check
    assert get_order(u) == order

    v, x = u, t
    while v.shape[0] > 1 and v.shape[1] > 1 and v.shape[2] > 1:
        v = v[::s1, ::s2, ::s3]
        dpex.call_kernel(change_values_3d, Range(*v.shape), v, k)

        x = x[::s1, ::s2, ::s3]
        change_values_3d_func(x, k)

        # check the value of the array view
        assert np.all(dpnp.asnumpy(v) == x)
        # check the value of the original arrays
        assert np.all(dpnp.asnumpy(u) == t)
