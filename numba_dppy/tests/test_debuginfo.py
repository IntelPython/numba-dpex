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
import os

import dpctl
import numpy as np
from numba import config
from numba.core import types

import numba_dppy as dppy
from numba_dppy import compiler
from numba_dppy.tests._helper import skip_test


debug_options = [True, False]


@pytest.fixture(params=debug_options)
def debug_option(request):
    return request.param


def get_kernel_ir(sycl_queue, fn, sig, debug=False):
    kernel = compiler.compile_kernel(sycl_queue, fn.py_func, sig, None, debug=debug)
    return kernel.assembly


def make_check(ir, val_to_search):
    """
    Check the compiled assembly for debuginfo.
    """

    m = re.search(val_to_search, ir, re.I)
    got = m is not None
    return got


def test_debug_flag_generates_ir_with_debuginfo(offload_device, debug_option):
    """
    Check debug info is emitting to IR if debug parameter is set to True
    """

    if skip_test(offload_device):
        pytest.skip()

    if offload_device in "level_zero:gpu:0":
        pytest.xfail("Failing compilation: SyclProgramCompilationError")

    @dppy.kernel
    def foo(x):
        return x

    with dpctl.device_context(offload_device) as sycl_queue:
        sig = (types.int32,)
        kernel_ir = get_kernel_ir(sycl_queue, foo, sig, debug=debug_option)

        expect = debug_option
        got = make_check(kernel_ir, r"!dbg")

        assert expect == got


def test_debug_info_locals_vars_on_no_opt(offload_device):
    """
    Check llvm debug tag DILocalVariable is emitting to IR for all variables if debug parameter is set to True
    and optimization is O0
    """

    pytest.xfail(
        "Assertion Cast->getSrcTy()->getPointerAddressSpace() == SPIRAS_Generic"
    )

    if skip_test(offload_device):
        pytest.skip()

    @dppy.kernel
    def foo(var_a, var_b, var_c):
        i = dppy.get_global_id(0)
        var_c[i] = var_a[i] + var_b[i]

    ir_tag_var_a = r'\!DILocalVariable\(name: "var_a"'
    ir_tag_var_b = r'\!DILocalVariable\(name: "var_b"'
    ir_tag_var_c = r'\!DILocalVariable\(name: "var_c"'
    ir_tag_var_i = r'\!DILocalVariable\(name: "i"'

    ir_tags = (ir_tag_var_a, ir_tag_var_b, ir_tag_var_c, ir_tag_var_i)

    opt_curr_value = os.environ.get("NUMBA_OPT", 3)  # 3 is a default value
    config.OPT = 0  # All variables are available on no opt level

    with dpctl.device_context(offload_device) as sycl_queue:
        sig = (types.float32[:], types.float32[:], types.float32[:])
        kernel_ir = get_kernel_ir(sycl_queue, foo, sig, debug=True)

        expect = True  # Expect tag is emitted

        for tag in ir_tags:
            got = make_check(kernel_ir, tag)
            assert expect == got

    config.OPT = opt_curr_value  # Return to the previous value


def test_debug_kernel_local_vars_in_ir(offload_device):
    """
    Check llvm debug tag DILocalVariable is emitting to IR for variables created in kernel
    """

    if skip_test(offload_device):
        pytest.skip()

    @dppy.kernel
    def foo(arr):
        index = dppy.get_global_id(0)
        local_d = 9 * 99 + 5
        arr[index] = local_d + 100

    ir_tag_var_index = r'\!DILocalVariable\(name: "index"'
    ir_tag_var_local_d = r'\!DILocalVariable\(name: "local_d"'

    ir_tags = (ir_tag_var_index, ir_tag_var_local_d)

    with dpctl.device_context(offload_device) as sycl_queue:
        sig = (types.float32[:],)
        kernel_ir = get_kernel_ir(sycl_queue, foo, sig, debug=True)

        expect = True  # Expect tag is emitted

        for tag in ir_tags:
            got = make_check(kernel_ir, tag)
            assert expect == got
