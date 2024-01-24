# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np
import pytest

import numba_dpex as dpex
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.tests._helper import get_all_dtypes

global_size = 100
N = global_size


list_of_dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)


@pytest.fixture(params=list_of_dtypes)
def return_dtype(request):
    return request.param


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    def _inpute_arrays():
        a = np.array([0], request.param)
        return a, request.param

    return _inpute_arrays


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


def test_kernel_atomic_simple(input_arrays, kernel_result_pair):
    a, dtype = input_arrays()
    kernel, expected = kernel_result_pair
    dpex.call_kernel(kernel, dpex.Range(global_size), a)
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


def test_kernel_atomic_local(input_arrays, return_list_of_op):
    a, dtype = input_arrays()
    op_type, expected = return_list_of_op
    f = get_func_local(op_type, dtype)
    kernel = dpex.kernel(f)
    dpex.call_kernel(kernel, dpex.NdRange(dpex.Range(N), dpex.Range(N)), a)
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


def test_kernel_atomic_multi_dim(
    return_list_of_op, return_list_of_dim, return_dtype
):
    op_type, expected = return_list_of_op
    dim = return_list_of_dim
    kernel = get_kernel_multi_dim(op_type, len(dim))
    a = np.zeros(dim, dtype=return_dtype)
    dpex.call_kernel(kernel, dpex.Range(global_size), a)
    assert a[0] == expected


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
@pytest.mark.parametrize(
    "dtype",
    get_all_dtypes(
        no_bool=True,
        no_int=True,
        no_float16=True,
        no_none=True,
        no_complex=True,
    ),
)
def test_atomic_fp_native(
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

    kernel.compile(
        args=argtypes,
        debug=False,
        compile_flags=None,
        target_ctx=dpex_kernel_target.target_context,
        typing_ctx=dpex_kernel_target.typing_context,
    )

    # TODO: this may fail if code is generated for platform that emulates atomic support?
    assert expected_spirv_function in kernel._llvm_module
