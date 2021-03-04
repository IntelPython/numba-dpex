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

import numpy as np
import numba_dppy as dppy
import pytest
from numba_dppy.context import device_context
from numba_dppy.tests.skip_tests import skip_test

list_of_filter_strs = [
    "opencl:gpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


@pytest.mark.xfail
def test_print_only_str(filter_str):
    try:
        with device_context(filter_str):
            pass
    except Exception:
        pytest.skip()

    @dppy.kernel
    def f():
        print("test")

    # This test will fail, we currently can not print only string.
    # The LLVM generated for printf() function with only string gets
    # replaced by a puts() which fails due to lack of addrspace in the
    # puts function signature right now, and would fail in general due
    # to lack of support for puts() in OpenCL.
    with device_context(filter_str), captured_stdout() as stdout:
        f[3, dppy.DEFAULT_LOCAL_SIZE]()


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


def test_print(filter_str, input_arrays, capfd):
    if skip_test(filter_str):
        pytest.skip()

    @dppy.kernel
    def f(a):
        print("test", a[0])

    a = input_arrays
    global_size = 3

    with device_context(filter_str):
        f[global_size, dppy.DEFAULT_LOCAL_SIZE](a)
        captured = capfd.readouterr()
        assert "test" in captured.out
