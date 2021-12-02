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

import dpctl
import numpy as np
import pytest

import numba_dppy as dppy
from numba_dppy.tests._helper import filter_strings


def skip(filter_str):
    if filter_str != "opencl:gpu:0":
        pytest.skip("Not tested")


@pytest.mark.xfail
@pytest.mark.parametrize("filter_str", filter_strings)
def test_print_only_str(filter_str):
    skip(filter_str)
    try:
        device = dpctl.SyclDevice(filter_str)
        with dpctl.device_context(device):
            pass
    except Exception:
        pytest.skip()

    @dppy.kernel
    def f():
        print("test", "test2")

    # This test will fail, we currently can not print only string.
    # The LLVM generated for printf() function with only string gets
    # replaced by a puts() which fails due to lack of addrspace in the
    # puts function signature right now, and would fail in general due
    # to lack of support for puts() in OpenCL.

    with dpctl.device_context(filter_str):
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


@pytest.mark.parametrize("filter_str", filter_strings)
def test_print(filter_str, input_arrays, capfd):
    skip(filter_str)

    @dppy.kernel
    def f(a):
        print("test", a[0])

    a = input_arrays
    global_size = 3

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        f[global_size, dppy.DEFAULT_LOCAL_SIZE](a)
        captured = capfd.readouterr()
        assert "test" in captured.out
