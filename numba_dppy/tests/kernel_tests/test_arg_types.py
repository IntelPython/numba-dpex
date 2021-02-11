import numpy as np
import numba_dppy as dppy
import pytest
import dpctl

global_size = 1054
local_size = 1
N = global_size * local_size

def mul_kernel(a, b, c):
    i = dppy.get_global_id(0)
    b[i] = a[i] * c

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
    a = np.array(np.random.random(N), request.param)
    b = np.empty_like(a, request.param)
    c = np.array([2], request.param)
    return a, b, c[0]


def test_kernel_arg_types(filter_str, input_arrays):
    try:
        with dpctl.device_context(filter_str):
            pass
    except Exception:
        pytest.skip()

    kernel = dppy.kernel(mul_kernel)
    a, actual, c = input_arrays
    expected = a * c
    with dpctl.device_context(filter_str):
        kernel[global_size, local_size](a, actual, c)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


def check_bool_kernel(A, test):
    if test:
        A[0] = 111
    else:
        A[0] = 222

def test_bool_type(filter_str):
    try:
        with dpctl.device_context(filter_str):
            pass
    except Exception:
        pytest.skip()

    kernel = dppy.kernel(check_bool_kernel)
    a = np.array([2], np.int64)

    with dpctl.device_context(filter_str):
        kernel[a.size, dppy.DEFAULT_LOCAL_SIZE](a, True)
        assert(a[0] == 111)
        kernel[a.size, dppy.DEFAULT_LOCAL_SIZE](a, False)
        assert(a[0] == 222)

