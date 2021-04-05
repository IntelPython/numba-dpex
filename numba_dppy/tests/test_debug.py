#! /usr/bin/env python
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

import re
import pytest

import numba_dppy as dppy
import dpctl

from numba.core import types, compiler
from numba_dppy import compiler
from numba_dppy.tests.skip_tests import skip_test


# TODO: Add level0 and move to common place
offload_devices = [
    "opencl:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=offload_devices)
def offload_device(request):
    return request.param


def get_kernel_ir(fn, sig, debug=False):
    kernel = compiler.compile_kernel(fn.sycl_queue, fn.py_func, sig, None, debug=debug)
    return kernel.assembly


def make_check(fn, sig, expect):
    """
    Check the compiled assembly for debuginfo.
    """
    ir = get_kernel_ir(fn, sig=sig, debug=expect)

    # Checking whether debug symbols have been emmited to IR
    m = re.search(r"!dbg", ir, re.I)
    got = m is not None
    assert expect == got


def test_debuginfo_in_ir(offload_device):
    """
    Check debug info is emitting to IR if debug parameter is set to True
    """

    if skip_test(offload_device):
        pytest.skip()

    debug_expect = True

    @dppy.kernel
    def foo(x):
        return x

    with dpctl.device_context(offload_device):
        sig = (types.int32,)
        make_check(foo, sig, debug_expect)


def test_debuginfo_not_in_ir(offload_device):
    """
    Check debug info is not emitting to IR if debug parameter is set to True
    """

    if skip_test(offload_device):
        pytest.skip()

    debug_expect = False

    @dppy.kernel
    def foo(x):
        return x

    with dpctl.device_context(offload_device):
        sig = (types.int32,)
        make_check(foo, sig, debug_expect)
