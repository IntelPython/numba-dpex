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
from numba_dppy.tests._helper import skip_test


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


def test_debug_flag_generates_ir_with_debuginfo_for_func(offload_device):
    """
    Check debug info is emitting to IR if debug parameter is set to True
    """

    if skip_test(offload_device):
        pytest.skip()

    if offload_device in "level_zero:gpu:0":
        pytest.xfail("Failing compilation: SyclProgramCompilationError")

    @dppy.func(debug=True)
    def func_sum(a, b):
        result = a + b
        return result

    @dppy.kernel(debug=True)
    def data_parallel_sum(a, b, c):
        i = dppy.get_global_id(0)
        c[i] = func_sum(a[i], b[i])

    ir_tag_func_sum = r'\!DISubprogram\(name: "test_debug_flag_generates_ir_with_debuginfo_for_func.<locals>.func_sum"'
    ir_tag_data_parallel_sum = r'\!DISubprogram\(name: "test_debug_flag_generates_ir_with_debuginfo_for_func.<locals>.data_parallel_sum"'
    ir_tags = (ir_tag_func_sum, ir_tag_data_parallel_sum)

    with dpctl.device_context(offload_device) as sycl_queue:
        sig = (
            types.float32[:],
            types.float32[:],
            types.float32[:],
        )
        kernel_ir = get_kernel_ir(sycl_queue, data_parallel_sum, sig, debug=True)

    for tag in ir_tags:
        got = make_check(kernel_ir, tag)
        assert got == True


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

    ir_tag_func_sum = r'\!DISubprogram\(name: "test_env_var_generates_ir_with_debuginfo_for_func.<locals>.func_sum"'
    ir_tag_data_parallel_sum = r'\!DISubprogram\(name: "test_env_var_generates_ir_with_debuginfo_for_func.<locals>.data_parallel_sum"'
    ir_tags = (ir_tag_func_sum, ir_tag_data_parallel_sum)

    dppy.compiler.DEBUGINFO = 1

    with dpctl.device_context(offload_device) as sycl_queue:
        sig = (
            types.float32[:],
            types.float32[:],
            types.float32[:],
        )
        kernel_ir = get_kernel_ir(sycl_queue, data_parallel_sum, sig, debug=True)

    dppy.compiler.DEBUGINFO = 0

    for tag in ir_tags:
        got = make_check(kernel_ir, tag)
        assert got == True
