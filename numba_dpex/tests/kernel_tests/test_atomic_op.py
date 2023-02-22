# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex import config
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.tests._helper import filter_strings, override_config

global_size = 100
N = global_size


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
    op = getattr(dpex.atomic, request.param[0])

    def f(a):
        op(a, 0, 1)

    return dpex.kernel(f), request.param[1]


skip_no_atomic_support = pytest.mark.skipif(
    not dpex.ocl.atomic_support_present(),
    reason="No atomic support",
)


@pytest.mark.parametrize("filter_str", filter_strings)
@skip_no_atomic_support
def test_kernel_atomic_simple(filter_str, input_arrays, kernel_result_pair):
    a, dtype = input_arrays
    kernel, expected = kernel_result_pair
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        kernel[global_size, dpex.DEFAULT_LOCAL_SIZE](a)
    assert a[0] == expected


def get_func_global(op_type, dtype):
    """Generate function for global address space

    Used as `generator(op_type, dtype)`.
    """
    op = getattr(dpex.atomic, op_type)

    def f(a):
        op(a, 0, 1)

    return f


def get_func_local(op_type, dtype):
    """Generate function for local address space

    Used as `generator(op_type, dtype)`.
    """
    op = getattr(dpex.atomic, op_type)

    def f(a):
        lm = dpex.local.array(1, dtype)
        lm[0] = a[0]
        dpex.barrier(dpex.GLOBAL_MEM_FENCE)
        op(lm, 0, 1)
        dpex.barrier(dpex.GLOBAL_MEM_FENCE)
        a[0] = lm[0]

    return f


@pytest.mark.parametrize("filter_str", filter_strings)
@skip_no_atomic_support
def test_kernel_atomic_local(filter_str, input_arrays, return_list_of_op):
    a, dtype = input_arrays
    op_type, expected = return_list_of_op
    f = get_func_local(op_type, dtype)
    kernel = dpex.kernel(f)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        gs = (N,)
        ls = (N,)
        kernel[gs, ls](a)
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
    op = getattr(dpex.atomic, op_type)
    if size == 1:
        idx = 0
    else:
        idx = (0,)
        for i in range(size - 1):
            idx += (0,)

    def f(a):
        op(a, idx, 1)

    return dpex.kernel(f)


@pytest.mark.parametrize("filter_str", filter_strings)
@skip_no_atomic_support
def test_kernel_atomic_multi_dim(
    filter_str, return_list_of_op, return_list_of_dim, return_dtype
):
    op_type, expected = return_list_of_op
    dim = return_list_of_dim
    kernel = get_kernel_multi_dim(op_type, len(dim))
    a = np.zeros(dim, return_dtype)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        kernel[global_size, dpex.DEFAULT_LOCAL_SIZE](a)
    assert a[0] == expected


skip_NATIVE_FP_ATOMICS_0 = pytest.mark.skipif(
    not config.NATIVE_FP_ATOMICS, reason="Native FP atomics disabled"
)


def skip_if_disabled(*args):
    return pytest.param(*args, marks=skip_NATIVE_FP_ATOMICS_0)


@skip_no_atomic_support
@pytest.mark.parametrize(
    "NATIVE_FP_ATOMICS, expected_native_atomic_for_device",
    [
        skip_if_disabled(1, lambda device: device != "opencl:cpu:0"),
        (0, lambda device: False),
    ],
)
@pytest.mark.parametrize(
    "function_generator", [get_func_global, get_func_local]
)
@pytest.mark.parametrize(
    "operator_name, expected_spirv_function",
    [
        ("add", "__spirv_AtomicFAddEXT"),
        ("sub", "__spirv_AtomicFAddEXT"),
    ],
)
@pytest.mark.parametrize("dtype", list_of_f_dtypes)
def test_atomic_fp_native(
    NATIVE_FP_ATOMICS,
    expected_native_atomic_for_device,
    function_generator,
    operator_name,
    expected_spirv_function,
    dtype,
):
    function = function_generator(operator_name, dtype)
    kernel = dpex.core.kernel_interface.spirv_kernel.SpirvKernel(
        function, function.__name__
    )
    args = [np.array([0], dtype)]
    argtypes = [
        dpex_kernel_target.typing_context.resolve_argument_type(arg)
        for arg in args
    ]

    with override_config("NATIVE_FP_ATOMICS", NATIVE_FP_ATOMICS):
        kernel.compile(
            args=argtypes,
            debug=False,
            compile_flags=None,
            target_ctx=dpex_kernel_target.target_context,
            typing_ctx=dpex_kernel_target.typing_context,
        )

        is_native_atomic = expected_spirv_function in kernel._llvm_module
        assert is_native_atomic == expected_native_atomic_for_device(
            dpctl.select_default_device().filter_string
        )
