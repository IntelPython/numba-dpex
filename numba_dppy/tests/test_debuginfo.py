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


def get_kernel_ir(sycl_queue, fn, sig, debug=None):
    kernel = compiler.compile_kernel(sycl_queue, fn.py_func, sig, None, debug=debug)
    return kernel.assembly


def make_check(ir, val_to_search):
    """
    Check the compiled assembly for debuginfo.
    """

    m = re.search(val_to_search, ir, re.I)
    got = m is not None
    return got


def test_debug_flag_generates_ir_with_debuginfo(debug_option):
    """
    Check debug info is emitting to IR if debug parameter is set to True
    """

    @dppy.kernel
    def foo(x):
        return x

    sycl_queue = dpctl.get_current_queue()
    sig = (types.int32,)

    kernel_ir = get_kernel_ir(sycl_queue, foo, sig, debug=debug_option)

    tag = "!dbg"

    if debug_option:
        assert tag in kernel_ir
    else:
        assert tag not in kernel_ir


def test_debug_info_locals_vars_on_no_opt():
    """
    Check llvm debug tag DILocalVariable is emitting to IR for all variables if debug parameter is set to True
    and optimization is O0
    """

    pytest.xfail(
        "Assertion Cast->getSrcTy()->getPointerAddressSpace() == SPIRAS_Generic"
    )

    @dppy.kernel
    def foo(var_a, var_b, var_c):
        i = dppy.get_global_id(0)
        var_c[i] = var_a[i] + var_b[i]

    ir_tags = [
        '!DILocalVariable(name: "var_a"',
        '!DILocalVariable(name: "var_b"',
        '!DILocalVariable(name: "var_c"',
        '!DILocalVariable(name: "i"',
    ]

    config.OPT = 0  # All variables are available on no opt level

    sycl_queue = dpctl.get_current_queue()
    sig = (types.float32[:], types.float32[:], types.float32[:])

    kernel_ir = get_kernel_ir(sycl_queue, foo, sig, debug=True)

    for tag in ir_tags:
        assert tag in kernel_ir

    config.OPT = 3  # Return to the default value


def test_debug_kernel_local_vars_in_ir():
    """
    Check llvm debug tag DILocalVariable is emitting to IR for variables created in kernel
    """

    @dppy.kernel
    def foo(arr):
        index = dppy.get_global_id(0)
        local_d = 9 * 99 + 5
        arr[index] = local_d + 100

    ir_tags = ['!DILocalVariable(name: "index"', '!DILocalVariable(name: "local_d"']

    sycl_queue = dpctl.get_current_queue()
    sig = (types.float32[:],)

    kernel_ir = get_kernel_ir(sycl_queue, foo, sig, debug=True)

    for tag in ir_tags:
        assert tag in kernel_ir


def test_debug_flag_generates_ir_with_debuginfo_for_func(debug_option):
    """
    Check debug info is emitting to IR if debug parameter is set to True
    """

    @dppy.func(debug=debug_option)
    def func_sum(a, b):
        result = a + b
        return result

    @dppy.kernel(debug=debug_option)
    def data_parallel_sum(a, b, c):
        i = dppy.get_global_id(0)
        c[i] = func_sum(a[i], b[i])

    ir_tags = [
        r'\!DISubprogram\(name: ".*func_sum"',
        r'\!DISubprogram\(name: ".*data_parallel_sum"',
    ]

    sycl_queue = dpctl.get_current_queue()
    sig = (
        types.float32[:],
        types.float32[:],
        types.float32[:],
    )

    kernel_ir = get_kernel_ir(sycl_queue, data_parallel_sum, sig, debug=debug_option)

    for tag in ir_tags:
        assert debug_option == make_check(kernel_ir, tag)


def test_env_var_generates_ir_with_debuginfo_for_func(offload_device):
    """
    Check debug info is emitting to IR if NUMBA_DPPY_DEBUGINFO is set to 1
    """

    if skip_test(offload_device):
        pytest.skip()

    if offload_device in "level_zero:gpu:0":
        pytest.xfail("Failing compilation: SyclProgramCompilationError")

    @dppy.func
    def func_sum(a, b):
        result = a + b
        return result

    @dppy.kernel
    def data_parallel_sum(a, b, c):
        i = dppy.get_global_id(0)
        c[i] = func_sum(a[i], b[i])

    ir_tag_func_sum = r'\!DISubprogram\(name: ".*func_sum"'
    ir_tag_data_parallel_sum = r'\!DISubprogram\(name: ".*data_parallel_sum"'
    ir_tags = (ir_tag_func_sum, ir_tag_data_parallel_sum)

    OLD_DEBUGINFO_DEFAULT = dppy.compiler.DEBUGINFO_DEFAULT
    dppy.compiler.DEBUGINFO_DEFAULT = 1

    with dpctl.device_context(offload_device) as sycl_queue:
        sig = (
            types.float32[:],
            types.float32[:],
            types.float32[:],
        )
        kernel_ir = get_kernel_ir(sycl_queue, data_parallel_sum, sig, debug=True)

    dppy.compiler.DEBUGINFO_DEFAULT = OLD_DEBUGINFO_DEFAULT

    for tag in ir_tags:
        got = make_check(kernel_ir, tag)
        assert got == True
