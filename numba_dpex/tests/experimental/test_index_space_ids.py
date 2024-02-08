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
from numba_dpex.kernel_api import Item, NdItem
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
