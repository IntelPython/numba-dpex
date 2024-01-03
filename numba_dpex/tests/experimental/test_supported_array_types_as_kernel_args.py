# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests different input array type support for the kernel."""

import dpctl.tensor as dpt
import dpnp
import pytest

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.tests._helper import get_all_dtypes

list_of_dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)

zeros_func = (dpt.zeros, dpnp.zeros)

_SIZE = 10


@pytest.fixture(params=((a, b) for a in zeros_func for b in list_of_dtypes))
def input_array(request):
    zeros, dtype = request.param
    return zeros(_SIZE, dtype=dtype)


@dpex_exp.kernel
def set_ones(a):
    i = dpex.get_global_id(0)
    a[i] = 1


def test_fetch_add(input_array):
    dpex_exp.call_kernel(set_ones, dpex.Range(_SIZE), input_array)

    assert input_array[0] == 1
