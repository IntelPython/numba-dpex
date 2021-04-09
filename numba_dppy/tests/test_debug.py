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

import dpctl
from numba.core import types

import numba_dppy as dppy
from numba_dppy import compiler
from numba_dppy.tests.skip_tests import skip_test


debug_options = [True, False]


@pytest.fixture(params=debug_options)
def debug_option(request):
    return request.param


def get_kernel_ir(fn, sig, debug=False):
    kernel = compiler.compile_kernel(fn.sycl_queue, fn.py_func, sig, None, debug=debug)
    return kernel.assembly


def make_check(ir):
    """
    Check the compiled assembly for debuginfo.
    """

    m = re.search(r"!dbg", ir, re.I)
    got = m is not None
    return got


def test_debug_flag_generates_ir_with_debuginfo(offload_device, debug_option):
    """
    Check debug info is emitting to IR if debug parameter is set to True
    """

    if skip_test(offload_device):
        pytest.skip()

    if offload_device in "level0:gpu:0":
        pytest.xfail("Failing compilation: SyclProgramCompilationError")

    @dppy.kernel
    def foo(x):
        return x

    with dpctl.device_context(offload_device):
        sig = (types.int32,)
        kernel_ir = get_kernel_ir(foo, sig, debug=debug_option)

        expect = debug_option
        got = make_check(kernel_ir)

        assert expect == got
