# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import filter_strings_opencl_gpu


@pytest.mark.parametrize("filter_str", filter_strings_opencl_gpu)
@pytest.mark.xfail
def test_print_only_str(filter_str):
    try:
        device = dpctl.SyclDevice(filter_str)
        with dpctl.device_context(device):
            pass
    except Exception:
        pytest.skip()

    @dpex.kernel
    def f():
        print("test", "test2")

    # This test will fail, we currently can not print only string.
    # The LLVM generated for printf() function with only string gets
    # replaced by a puts() which fails due to lack of addrspace in the
    # puts function signature right now, and would fail in general due
    # to lack of support for puts() in OpenCL.

    with dpctl.device_context(filter_str):
        f[3, dpex.DEFAULT_LOCAL_SIZE]()


list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = np.array([0], request.param)
    a[0] = 6
    return a


@pytest.mark.parametrize("filter_str", filter_strings_opencl_gpu)
def test_print(filter_str, input_arrays, capfd):
    @dpex.kernel
    def f(a):
        print("test", a[0])

    a = input_arrays
    global_size = 3

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        f[global_size, dpex.DEFAULT_LOCAL_SIZE](a)
        captured = capfd.readouterr()
        assert "test" in captured.out
