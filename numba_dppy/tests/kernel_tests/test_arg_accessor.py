import numpy as np

import numba_dppy as dppy
import pytest
import dpctl


def call_kernel(global_size, local_size, A, B, C, func):
    func[global_size, local_size](A, B, C)


global_size = 10
local_size = 1
N = global_size * local_size


def sum_kernel(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


list_of_filter_strs = [
    "opencl:gpu:0",
    "level0:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    c = np.zeros_like(a)
    return a, b, c


list_of_kernel_opt = [
    {"read_only": ["a", "b"], "write_only": ["c"], "read_write": []},
    {},
]


@pytest.fixture(params=list_of_kernel_opt)
def kernel(request):
    return dppy.kernel(access_types=request.param)(sum_kernel)


def test_kernel_arg_accessor(filter_str, input_arrays, kernel):
    try:
        with dpctl.device_context(filter_str):
            pass
    except Exception:
        pytest.skip()

    a, b, actual = input_arrays
    expected = a + b
    with dpctl.device_context(filter_str):
        call_kernel(global_size, local_size, a, b, actual, kernel)
