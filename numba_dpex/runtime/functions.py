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

from ctypes import PYFUNCTYPE, Structure, c_int, c_int64, c_void_p, py_object

from . import _rt_python


class SyclUsmArrayStruct(Structure):
    """Corresponds to dp_arystruct_t"""

    _fields_ = [
        ("meminfo", c_void_p),  # void *meminfo;
        ("parent", c_void_p),  # PyObject *parent;
        ("nitems", c_int64),  # npy_intp nitems;
        ("itemsize", c_int64),  # npy_intp itemsize;
        ("data", c_void_p),  # void *data;
        ("syclobj", c_void_p),  # PyObject *syclobj;
        ("shape_and_strides", c_int64 * 10),  # npy_intp shape_and_strides[];
    ]


DPEX_RT_sycl_usm_array_from_python = PYFUNCTYPE(c_int, py_object, c_void_p)(
    _rt_python.DPEX_RT_sycl_usm_array_from_python
)

PySyclUsmArray_Check = PYFUNCTYPE(c_int, py_object)(
    _rt_python.PySyclUsmArray_Check
)

PySyclUsmArray_NDIM = PYFUNCTYPE(c_int, py_object)(
    _rt_python.PySyclUsmArray_NDIM
)

itemsize_from_typestr = PYFUNCTYPE(c_int, py_object)(
    _rt_python.itemsize_from_typestr
)
