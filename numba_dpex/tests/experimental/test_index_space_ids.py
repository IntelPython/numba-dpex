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

_SIZE = 10


@dpex_exp.kernel
def set_ones_no_item(a):
    a[0] = 1


@dpex_exp.kernel
def set_ones_item(item: Item, a):
    i = item.get_id(0)
    a[i] = 1


@dpex_exp.kernel
def set_ones_nd_item(nd_item: NdItem, a):
    i = nd_item.get_global_id(0)
    a[i] = 1


def test_item_get_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(set_ones_item, dpex.Range(a.size), a)

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.float32))


# TODO: https://github.com/IntelPython/numba-dpex/issues/1308
@skip_windows
def test_nd_item_get_global_id():
    a = dpnp.zeros(_SIZE, dtype=dpnp.float32)
    dpex_exp.call_kernel(
        set_ones_nd_item, dpex.NdRange((a.size,), (a.size,)), a
    )

    assert np.array_equal(a.asnumpy(), np.ones(a.size, dtype=np.float32))


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
