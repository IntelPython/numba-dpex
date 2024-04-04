# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from numba import float32, float64, int32, int64, uint32, uint64
from numba.core import types

import numba_dpex.core.utils.itanium_mangler as itanium_mangler
from numba_dpex.kernel_api import AddressSpace as address_space

list_of_dtypes = [
    (int32, "i"),
    (int64, "x"),
    (uint32, "j"),
    (uint64, "y"),
    (float32, "f"),
    (float64, "d"),
]


@pytest.fixture(params=list_of_dtypes)
def dtypes(request):
    return request.param


list_of_addrspaces = [
    (address_space.PRIVATE.value, "3AS0"),
    (address_space.GLOBAL.value, "3AS1"),
    (address_space.LOCAL.value, "3AS3"),
    (address_space.GENERIC.value, "3AS4"),
]


@pytest.fixture(params=list_of_addrspaces)
def addrspaces(request):
    return request.param


def test_mangling_arg_type(dtypes):
    dtype, expected_str = dtypes
    got = itanium_mangler.mangle_type(types.CPointer(dtype))
    expected = "P" + expected_str
    assert got == expected


def test_mangling_arg_type_2(dtypes, addrspaces):
    dtype, expected_dtype_str = dtypes
    addrspace, expected_addrspace_str = addrspaces
    got = itanium_mangler.mangle_type(
        types.CPointer(dtype, addrspace=addrspace)
    )
    expected = "PU" + expected_addrspace_str + expected_dtype_str
    assert got == expected
