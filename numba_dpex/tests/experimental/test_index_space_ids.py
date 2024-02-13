# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests different input array type support for the kernel."""

import dpnp
import numpy as np
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.kernel_api import Item, NdItem, NdRange
from numba_dpex.kernel_api import call_kernel as kapi_call_kernel
from numba_dpex.tests._helper import skip_windows

_SIZE = 16
_GROUP_SIZE = 4


@dpex_exp.kernel
def set_ones_no_item(a):
    a[0] = 1


@dpex_exp.kernel
def set_ones_item(item: Item, a):
    i = item.get_id(0)
    a[i] = 1


@dpex_exp.kernel
def set_last_one_item(item: Item, a):
    i = item.get_range(0) - 1
    a[i] = 1


@dpex_exp.kernel
def set_last_one_nd_item(item: NdItem, a):
    if item.get_global_id(0) == 0:
        i = item.get_global_range(0) - 1
        a[0] = i
        a[i] = 1


@dpex_exp.kernel
def set_last_group_one_nd_item(item: NdItem, a):
    if item.get_global_id(0) == 0:
        i = item.get_local_range(0) - 1
        a[0] = i
        a[i] = 1


@dpex_exp.kernel
def set_ones_nd_item(nd_item: NdItem, a):
    i = nd_item.get_global_id(0)
    a[i] = 1


@dpex_exp.kernel
def set_local_ones_nd_item(nd_item: NdItem, a):
    i = nd_item.get_local_id(0)
    a[i] = 1


def _get_group_id_driver(nditem: NdItem, a):
    i = nditem.get_global_id(0)
    g = nditem.get_group()
    a[i] = g.get_group_id(0)


def _get_group_range_driver(nditem: NdItem, a):
    i = nditem.get_global_id(0)
    g = nditem.get_group()
    a[i] = g.get_group_range(0)


def _get_group_local_range_driver(nditem: NdItem, a):
    i = nditem.get_global_id(0)
    g = nditem.get_group()
    a[i] = g.get_local_range(0)


def test_item_get_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(set_ones_item, dpex.Range(a.size), a)

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.float32))


def test_item_get_range():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(set_last_one_item, dpex.Range(a.size), a)

    want = np.zeros(a.size, dtype=np.float32)
    want[-1] = 1

    assert np.array_equal(a.asnumpy(), want)


# TODO: https://github.com/IntelPython/numba-dpex/issues/1308
@skip_windows
def test_nd_item_get_global_range():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(
        set_last_one_nd_item, dpex.NdRange((a.size,), (_GROUP_SIZE,)), a
    )

    want = np.zeros(a.size, dtype=np.float32)
    want[-1] = 1
    want[0] = a.size - 1

    assert np.array_equal(a.asnumpy(), want)


# TODO: https://github.com/IntelPython/numba-dpex/issues/1308
@skip_windows
def test_nd_item_get_local_range():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(
        set_last_group_one_nd_item, dpex.NdRange((a.size,), (_GROUP_SIZE,)), a
    )

    want = np.zeros(a.size, dtype=np.float32)
    want[_GROUP_SIZE - 1] = 1
    want[0] = _GROUP_SIZE - 1

    assert np.array_equal(a.asnumpy(), want)


# TODO: https://github.com/IntelPython/numba-dpex/issues/1308
@skip_windows
def test_nd_item_get_global_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(
        set_ones_nd_item, dpex.NdRange((a.size,), (_GROUP_SIZE,)), a
    )

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.float32))


# TODO: https://github.com/IntelPython/numba-dpex/issues/1308
@skip_windows
def test_nd_item_get_local_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)

    dpex_exp.call_kernel(
        set_local_ones_nd_item, dpex.NdRange((a.size,), (_GROUP_SIZE,)), a
    )

    assert np.array_equal(
        a.asnumpy(),
        np.array(
            [1] * _GROUP_SIZE + [0] * (a.size - _GROUP_SIZE),
            dtype=np.float32,
        ),
    )


def test_error_item_get_global_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)

    with pytest.raises(TypingError):
        dpex_exp.call_kernel(set_ones_nd_item, dpex.Range(a.size), a)


def test_no_item():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(set_ones_no_item, dpex.Range(a.size), a)

    assert np.array_equal(
        a.asnumpy(), np.array([1] + [0] * (a.size - 1), dtype=np.float32)
    )


# TODO: https://github.com/IntelPython/numba-dpex/issues/1308
@skip_windows
def test_get_group_id():
    global_size = 100
    group_size = 20
    num_groups = global_size // group_size

    a = dpnp.empty(global_size, dtype=dpnp.int32)
    ka = dpnp.empty(global_size, dtype=dpnp.int32)
    expected = np.empty(global_size, dtype=np.int32)
    ndrange = NdRange((global_size,), (group_size,))
    dpex_exp.call_kernel(dpex_exp.kernel(_get_group_id_driver), ndrange, a)
    kapi_call_kernel(_get_group_id_driver, ndrange, ka)

    for gid in range(num_groups):
        for lid in range(group_size):
            expected[gid * group_size + lid] = gid

    assert np.array_equal(a.asnumpy(), expected)
    assert np.array_equal(ka.asnumpy(), expected)


# TODO: https://github.com/IntelPython/numba-dpex/issues/1308
@skip_windows
def test_get_group_range():
    global_size = 100
    group_size = 20
    num_groups = global_size // group_size

    a = dpnp.empty(global_size, dtype=dpnp.int32)
    ka = dpnp.empty(global_size, dtype=dpnp.int32)
    expected = np.empty(global_size, dtype=np.int32)
    ndrange = NdRange((global_size,), (group_size,))
    dpex_exp.call_kernel(dpex_exp.kernel(_get_group_range_driver), ndrange, a)
    kapi_call_kernel(_get_group_range_driver, ndrange, ka)

    for gid in range(num_groups):
        for lid in range(group_size):
            expected[gid * group_size + lid] = num_groups

    assert np.array_equal(a.asnumpy(), expected)
    assert np.array_equal(ka.asnumpy(), expected)


# TODO: https://github.com/IntelPython/numba-dpex/issues/1308
@skip_windows
def test_get_group_local_range():
    global_size = 100
    group_size = 20
    num_groups = global_size // group_size

    a = dpnp.empty(global_size, dtype=dpnp.int32)
    ka = dpnp.empty(global_size, dtype=dpnp.int32)
    expected = np.empty(global_size, dtype=np.int32)
    ndrange = NdRange((global_size,), (group_size,))
    dpex_exp.call_kernel(
        dpex_exp.kernel(_get_group_local_range_driver), ndrange, a
    )
    kapi_call_kernel(_get_group_local_range_driver, ndrange, ka)

    for gid in range(num_groups):
        for lid in range(group_size):
            expected[gid * group_size + lid] = group_size

    assert np.array_equal(a.asnumpy(), expected)
    assert np.array_equal(ka.asnumpy(), expected)


I_SIZE, J_SIZE, K_SIZE = 2, 3, 4


@dpex_exp.kernel
def set_3d_ones_item(item: Item, a):
    i = item.get_id(0)
    j = item.get_id(1)
    k = item.get_id(2)

    # Since we have different sizes for each dimention, wrong order will result
    # that some indexes will be set twice and some won't be set.
    index = i + I_SIZE * (j + J_SIZE * k)

    a[index] = 1


# TODO: CI tests failing for some reason... Works fine locally on cpu and gpu
@pytest.mark.skip
def test_index_order():
    a = dpnp.zeros(I_SIZE * J_SIZE * K_SIZE, dtype=dpnp.int32)

    dpex_exp.call_kernel(
        set_3d_ones_item, dpex.Range(I_SIZE, J_SIZE, K_SIZE), a
    )

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.int32))
