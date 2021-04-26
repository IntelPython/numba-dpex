# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numba_dppy as dppy
import pytest
import numba_dppy.extended_numba_itanium_mangler as itanium_mangler
from numba import int32, int64, uint32, uint64, float32, float64
from numba.core import types
from numba_dppy.target import (
    SPIR_PRIVATE_ADDRSPACE,
    SPIR_GLOBAL_ADDRSPACE,
    SPIR_CONSTANT_ADDRSPACE,
    SPIR_LOCAL_ADDRSPACE,
    SPIR_GENERIC_ADDRSPACE,
)

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
    (SPIR_PRIVATE_ADDRSPACE, "3AS0"),
    (SPIR_GLOBAL_ADDRSPACE, "3AS1"),
    (SPIR_CONSTANT_ADDRSPACE, "3AS2"),
    (SPIR_LOCAL_ADDRSPACE, "3AS3"),
    (SPIR_GENERIC_ADDRSPACE, "3AS4"),
]


@pytest.fixture(params=list_of_addrspaces)
def addrspaces(request):
    return request.param


def test_mangling_arg_type(dtypes):
    dtype, expected_str = dtypes
    got = itanium_mangler.mangle_type(types.CPointer(dtype))
    expected = "P" + expected_str
    assert got == expected


def test_mangling_arg_type(dtypes, addrspaces):
    dtype, expected_dtype_str = dtypes
    addrspace, expected_addrspace_str = addrspaces
    got = itanium_mangler.mangle_type(types.CPointer(dtype, addrspace=addrspace))
    expected = "PU" + expected_addrspace_str + expected_dtype_str
    assert got == expected
