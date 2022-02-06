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

import ctypes
from functools import reduce

import dpnp
import numpy
import pytest
from numpy import dtype

from numba_dppy import runtime
from numba_dppy.runtime import _rt_python


def test_pointers_exposed():
    def exposed(function_name):
        assert hasattr(_rt_python, function_name)
        assert isinstance(getattr(_rt_python, function_name), int)

    exposed("DPPY_RT_sycl_usm_array_from_python")
    exposed("DPPY_RT_sycl_usm_array_to_python_acqref")

    exposed("PySyclUsmArray_Check")
    exposed("PySyclUsmArray_NDIM")

    exposed("itemsize_from_typestr")


@pytest.mark.parametrize(
    "array",
    [
        dpnp.ndarray([10]),
        dpnp.ndarray([2, 3], dtype=numpy.int32),
    ],
)
def test_sycl_usm_array_interface_equals_array_struct(array):
    suai = array.__sycl_usm_array_interface__
    struct = runtime.SyclUsmArrayStruct()

    ret = runtime.DPPY_RT_sycl_usm_array_from_python(
        array, ctypes.byref(struct)
    )

    assert ret == 0
    # assert struct.meminfo == ?
    assert struct.parent == id(array)
    assert struct.nitems == reduce(lambda x, y: x * y, suai["shape"])
    assert struct.itemsize == dtype(suai["typestr"]).itemsize
    assert struct.data == suai["data"][0]
    assert struct.syclobj == id(suai["syclobj"])
    assert struct.shape_and_strides[: array.ndim] == list(suai["shape"])
    # TODO: support strides. They are zeroes now.
    assert struct.shape_and_strides[
        array.ndim : 2 * array.ndim
    ] == array.ndim * [0]


def test_PySyclUsmArray_Check():
    assert runtime.PySyclUsmArray_Check(dpnp.ndarray([10])) == 1
    assert runtime.PySyclUsmArray_Check(object()) == 0


def test_PySyclUsmArray_NDIM():
    assert runtime.PySyclUsmArray_NDIM(dpnp.ndarray([10])) == 1
    assert runtime.PySyclUsmArray_NDIM(dpnp.ndarray([2, 3])) == 2


@pytest.mark.parametrize("typestr", ["i4", "<i4", "|f8", "f4"])
def test_itemsize_from_typestr(typestr):
    expected = numpy.dtype(typestr).itemsize
    assert runtime.itemsize_from_typestr(typestr) == expected
