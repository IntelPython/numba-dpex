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

import os

import dpctl
import numpy as np
import pytest

import numba_dppy as dppy
from numba_dppy import config
from numba_dppy.tests._helper import override_config, skip_test

global_size = 100
N = global_size


list_of_filter_strs = [
    "opencl:gpu:0",
    "level_zero:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_i_dtypes = [
    np.int32,
    np.int64,
]

list_of_f_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_i_dtypes + list_of_f_dtypes)
def return_dtype(request):
    return request.param


@pytest.fixture(params=list_of_f_dtypes)
def fdtype(request):
    return request.param


@pytest.fixture(params=list_of_i_dtypes + list_of_f_dtypes)
def input_arrays(request):
    a = np.array([0], request.param)
    return a, request.param


list_of_op = [
    ("add", N),
    ("sub", -N),
]


@pytest.fixture(params=list_of_op)
def return_list_of_op(request):
    return request.param[0], request.param[1]


@pytest.fixture(params=list_of_op)
def kernel_result_pair(request):
    op = getattr(dppy.atomic, request.param[0])

    def f(a):
        op(a, 0, 1)

    return dppy.kernel(f), request.param[1]


def atomic_skip_test(device_type):
    skip = False
    if skip_test(device_type):
        skip = True

    if not skip:
        if not dppy.ocl.atomic_support_present():
            skip = True

    return skip


def test_kernel_atomic_simple(filter_str, input_arrays, kernel_result_pair):
    if atomic_skip_test(filter_str):
        pytest.skip()

    a, dtype = input_arrays
    kernel, expected = kernel_result_pair
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](a)
    assert a[0] == expected


def get_func_global(op_type, dtype):
    """Generate function for global address space

    Used as `generator(op_type, dtype)`.
    """
    op = getattr(dppy.atomic, op_type)

    def f(a):
        op(a, 0, 1)

    return f


def get_func_local(op_type, dtype):
    """Generate function for local address space

    Used as `generator(op_type, dtype)`.
    """
    op = getattr(dppy.atomic, op_type)

    def f(a):
        lm = dppy.local.array(1, dtype)
        lm[0] = a[0]
        dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
        op(lm, 0, 1)
        dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
        a[0] = lm[0]

    return f


def test_kernel_atomic_local(filter_str, input_arrays, return_list_of_op):
    if atomic_skip_test(filter_str):
        pytest.skip()

    a, dtype = input_arrays
    op_type, expected = return_list_of_op
    f = get_func_local(op_type, dtype)
    kernel = dppy.kernel(f)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        kernel[global_size, global_size](a)
    assert a[0] == expected


list_of_dim = [
    (1,),
    (1, 1),
    (1, 1, 1),
]


@pytest.fixture(params=list_of_dim)
def return_list_of_dim(request):
    return request.param


def get_kernel_multi_dim(op_type, size):
    op = getattr(dppy.atomic, op_type)
    if size == 1:
        idx = 0
    else:
        idx = (0,)
        for i in range(size - 1):
            idx += (0,)

    def f(a):
        op(a, idx, 1)

    return dppy.kernel(f)


def test_kernel_atomic_multi_dim(
    filter_str, return_list_of_op, return_list_of_dim, return_dtype
):
    if atomic_skip_test(filter_str):
        pytest.skip()

    op_type, expected = return_list_of_op
    dim = return_list_of_dim
    kernel = get_kernel_multi_dim(op_type, len(dim))
    a = np.zeros(dim, return_dtype)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](a)
    assert a[0] == expected


@pytest.mark.skipif(
    not config.NATIVE_FP_ATOMICS, reason="Native FP atomics disabled"
)
@pytest.mark.parametrize(
    "NATIVE_FP_ATOMICS, expected_native_atomic_for_device",
    [
        (1, lambda device: device != "opencl:cpu:0"),
        (0, lambda device: False),
    ],
)
@pytest.mark.parametrize(
    "function_generator", [get_func_global, get_func_local]
)
@pytest.mark.parametrize("operator_name", map(lambda x: x[0], list_of_op))
@pytest.mark.parametrize("dtype", list_of_f_dtypes)
def test_atomic_fp_native(
    filter_str,
    NATIVE_FP_ATOMICS,
    expected_native_atomic_for_device,
    function_generator,
    operator_name,
    dtype,
):
    if atomic_skip_test(filter_str):
        pytest.skip(f"No atomic support present for device {filter_str}")

    function = function_generator(operator_name, dtype)
    kernel = dppy.kernel(function)
    argtypes = kernel._get_argtypes(np.array([0], dtype))

    with override_config("NATIVE_FP_ATOMICS", NATIVE_FP_ATOMICS):

        with dpctl.device_context(filter_str) as sycl_queue:

            specialized_kernel = kernel[
                global_size, dppy.DEFAULT_LOCAL_SIZE
            ].specialize(argtypes, sycl_queue)

            is_native_atomic = (
                "__spirv_AtomicFAddEXT" in specialized_kernel.assembly
            )
            assert is_native_atomic == expected_native_atomic_for_device(
                filter_str
            )
