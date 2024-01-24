#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

import numba_dpex as dpex
from numba_dpex import float32, int32, usm_ndarray
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.tests._helper import override_config

debug_options = [True, False]

f32arrty = usm_ndarray(ndim=1, dtype=float32, layout="C")


@pytest.fixture(params=debug_options)
def debug_option(request):
    return request.param


def get_kernel_ir(fn, sig, debug=False):
    kernel = dpex.core.kernel_interface.spirv_kernel.SpirvKernel(
        fn, fn.__name__
    )
    kernel.compile(
        args=sig,
        target_ctx=dpex_kernel_target.target_context,
        typing_ctx=dpex_kernel_target.typing_context,
        debug=debug,
        compile_flags=None,
    )
    return kernel.llvm_module


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

    def foo(x):
        x = 1  # noqa

    sig = (int32,)
    kernel_ir = get_kernel_ir(foo, sig, debug=debug_option)
    tag = "!dbg"

    if debug_option:
        assert tag in kernel_ir
    else:
        assert tag not in kernel_ir


def test_debug_info_locals_vars_on_no_opt():
    """
    Check llvm debug tag DILocalVariable is emitting to IR for all variables
    if debug parameter is set to True and optimization is O0
    """

    def foo(var_a, var_b, var_c):
        i = dpex.get_global_id(0)
        var_c[i] = var_a[i] + var_b[i]

    ir_tags = [
        '!DILocalVariable(name: "var_a"',
        '!DILocalVariable(name: "var_b"',
        '!DILocalVariable(name: "var_c"',
        '!DILocalVariable(name: "i"',
    ]
    sig = (f32arrty, f32arrty, f32arrty)

    with override_config("OPT", 0):
        kernel_ir = get_kernel_ir(foo, sig, debug=True)

    for tag in ir_tags:
        assert tag in kernel_ir


def test_debug_kernel_local_vars_in_ir():
    """
    Check llvm debug tag DILocalVariable is emitting to IR for variables
    created in kernel
    """

    def foo(arr):
        index = dpex.get_global_id(0)
        local_d = 9 * 99 + 5
        arr[index] = local_d + 100

    ir_tags = [
        '!DILocalVariable(name: "index"',
        '!DILocalVariable(name: "local_d"',
    ]
    sig = (f32arrty,)
    kernel_ir = get_kernel_ir(foo, sig, debug=True)

    for tag in ir_tags:
        assert tag in kernel_ir


def test_debug_flag_generates_ir_with_debuginfo_for_func(debug_option):
    """
    Check debug info is emitting to IR if debug parameter is set to True
    """

    @dpex.func(debug=debug_option)
    def func_sum(a, b):
        result = a + b
        return result

    def data_parallel_sum(a, b, c):
        i = dpex.get_global_id(0)
        c[i] = func_sum(a[i], b[i])

    ir_tags = [
        r'\!DISubprogram\(name: ".*func_sum\$?\d*"',
        r'\!DISubprogram\(name: ".*data_parallel_sum\$?\d*"',
    ]

    sig = (f32arrty, f32arrty, f32arrty)

    kernel_ir = get_kernel_ir(data_parallel_sum, sig, debug=debug_option)

    for tag in ir_tags:
        assert debug_option == make_check(kernel_ir, tag)


def test_env_var_generates_ir_with_debuginfo_for_func(debug_option):
    """
    Check debug info is emitting to IR if NUMBA_DPEX_DEBUGINFO is set to 1
    """

    @dpex.func
    def func_sum(a, b):
        result = a + b
        return result

    def data_parallel_sum(a, b, c):
        i = dpex.get_global_id(0)
        c[i] = func_sum(a[i], b[i])

    ir_tags = [
        r'\!DISubprogram\(name: ".*func_sum\$?\d*"',
        r'\!DISubprogram\(name: ".*data_parallel_sum\$\d*"',
    ]

    sig = (f32arrty, f32arrty, f32arrty)

    with override_config("DEBUGINFO_DEFAULT", int(debug_option)):
        kernel_ir = get_kernel_ir(data_parallel_sum, sig)

    for tag in ir_tags:
        assert debug_option == make_check(kernel_ir, tag)


def test_debuginfo_DISubprogram_linkageName():
    def func(a, b):
        i = dpex.get_global_id(0)
        b[i] = a[i]

    ir_tags = [
        r'\!DISubprogram\(.*linkageName: ".*func.*"',
    ]

    sig = (f32arrty, f32arrty)
    kernel_ir = get_kernel_ir(func, sig, debug=True)

    for tag in ir_tags:
        assert make_check(kernel_ir, tag)


def test_debuginfo_DICompileUnit_language_and_producer():
    def func(a, b):
        i = dpex.get_global_id(0)
        b[i] = a[i]

    ir_tags = [
        r"\!DICompileUnit\(language: DW_LANG_C_plus_plus,",
    ]

    sig = (f32arrty, f32arrty)

    kernel_ir = get_kernel_ir(func, sig, debug=True)

    for tag in ir_tags:
        assert make_check(kernel_ir, tag)
