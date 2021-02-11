import numpy as np

import numba_dppy as dppy
import pytest
import dpctl

def call_kernel(global_size, local_size, A, B, C, func):
    func[global_size, local_size](A, B, C)

global_size = 100
local_size = 1
N = global_size * local_size


list_of_filter_strs = [
    "opencl:gpu:0",
    "level0:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = np.array([0], request.param)
    return a

list_of_op = [
    ("add", N),
    ("sub", -N),
]

@pytest.fixture(params=list_of_op)
def kernel_result_pair(request):
    op = getattr(dppy.atomic, request.param[0])
    def f(a):
        op(a, 0, 1)
    return dppy.kernel(f), request.param[1]


def test_kernel_atomic_simple(filter_str, input_arrays, kernel_result_pair):
    try:
        with dpctl.device_context(filter_str):
            pass
    except Exception:
        pytest.skip()

    a = input_arrays
    kernel, expected = kernel_result_pair
    with dpctl.device_context(filter_str):
        kernel[global_size, local_size](a)
    assert(a[0] == expected)

