#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
from numba.core import types

import numba_dpex as dpex
from numba_dpex import float32, int32, usm_ndarray
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.core.types.kernel_api.index_space_ids import ItemType
from numba_dpex.tests._helper import override_config

debug_options = [True, False]

f32arrty = usm_ndarray(ndim=1, dtype=float32, layout="C")
itemty = ItemType(ndim=1)


@pytest.fixture(params=debug_options)
def debug_option(request):
    return request.param


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

    def foo(item, x):
        i = item.get_id(0)
        x[i] = 1  # noqa

    sig = (itemty, f32arrty)
    disp = dpex.kernel(sig, debug=debug_option)(foo)
    kcres = disp.get_compile_result(types.void(itemty, f32arrty))
    kernel_ir = kcres.library.get_llvm_str()

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

    def foo(item, var_a, var_b, var_c):
        i = item.get_id(0)
        var_c[i] = var_a[i] + var_b[i]

    ir_tags = [
        '!DILocalVariable(name: "var_a"',
        '!DILocalVariable(name: "var_b"',
        '!DILocalVariable(name: "var_c"',
        '!DILocalVariable(name: "i"',
    ]
    sig = (itemty, f32arrty, f32arrty, f32arrty)

    with override_config("OPT", 0):
        disp = dpex.kernel(sig, debug=True)(foo)
        kcres = disp.get_compile_result(
            types.void(itemty, f32arrty, f32arrty, f32arrty)
        )
        kernel_ir = kcres.library.get_llvm_str()

    for tag in ir_tags:
        assert tag in kernel_ir


def test_debug_kernel_local_vars_in_ir():
    """
    Check llvm debug tag DILocalVariable is emitting to IR for variables
    created in kernel
    """

    def foo(item, arr):
        index = item.get_id(0)
        local_d = 9 * 99 + 5
        arr[index] = local_d + 100

    ir_tags = [
        '!DILocalVariable(name: "index"',
        '!DILocalVariable(name: "local_d"',
    ]
    sig = (itemty, f32arrty)
    disp = dpex.kernel(sig, debug=True)(foo)
    kcres = disp.get_compile_result(types.void(itemty, f32arrty))
    kernel_ir = kcres.library.get_llvm_str()

    for tag in ir_tags:
        assert tag in kernel_ir


def test_debug_flag_generates_ir_with_debuginfo_for_func(debug_option):
    """
    Check debug info is emitting to IR if debug parameter is set to True
    """

    @dpex.device_func(debug=debug_option)
    def func_sum(a, b):
        result = a + b
        return result

    def data_parallel_sum(item, a, b, c):
        i = item.get_id(0)
        c[i] = func_sum(a[i], b[i])

    ir_tags = [
        r'\!DISubprogram\(name: ".*func_sum*"',
        r'\!DISubprogram\(name: ".*data_parallel_sum*"',
    ]

    sig = (itemty, f32arrty, f32arrty, f32arrty)
    disp = dpex.kernel(sig, debug=debug_option)(data_parallel_sum)
    kcres = disp.get_compile_result(
        types.void(itemty, f32arrty, f32arrty, f32arrty)
    )
    kernel_ir = kcres.library.get_llvm_str()

    for tag in ir_tags:
        assert debug_option == make_check(kernel_ir, tag)


def test_env_var_generates_ir_with_debuginfo_for_func(debug_option):
    """
    Check debug info is emitting to IR if NUMBA_DPEX_DEBUGINFO is set to 1
    """

    @dpex.device_func(debug=debug_option)
    def func_sum(a, b):
        result = a + b
        return result

    def data_parallel_sum(item, a, b, c):
        i = item.get_id(0)
        c[i] = func_sum(a[i], b[i])

    ir_tags = [
        r'\!DISubprogram\(name: ".*func_sum*"',
        r'\!DISubprogram\(name: ".*data_parallel_sum"',
    ]

    sig = (itemty, f32arrty, f32arrty, f32arrty)

    with override_config("DEBUGINFO_DEFAULT", int(debug_option)):
        disp = dpex.kernel(sig, debug=debug_option, inline_threshold=0)(
            data_parallel_sum
        )
        kcres = disp.get_compile_result(
            types.void(itemty, f32arrty, f32arrty, f32arrty)
        )
        kernel_ir = kcres.library.get_llvm_str()

    for tag in ir_tags:
        assert debug_option == make_check(kernel_ir, tag)


def test_debuginfo_DISubprogram_linkageName():
    """Tests to check that the linkagename tag is not set by numba-dpex."""

    def foo(item, a, b):
        i = item.get_id(0)
        b[i] = a[i]

    ir_tags = [
        r'\!DISubprogram\(.*linkageName: ".*foo.*"',
    ]

    sig = (itemty, f32arrty, f32arrty)
    disp = dpex.kernel(sig, debug=debug_option)(foo)
    kcres = disp.get_compile_result(types.void(itemty, f32arrty, f32arrty))
    kernel_ir = kcres.library.get_llvm_str()

    for tag in ir_tags:
        # Ensure that linkagename (DW_AT_linkagename) tag is not present for
        # the DISubprogram attribute.
        assert not make_check(kernel_ir, tag)


def test_debuginfo_DICompileUnit_language_and_producer():
    def foo(item, a, b):
        i = item.get_id(0)
        b[i] = a[i]

    ir_tags = [
        r"\!DICompileUnit\(language: DW_LANG_C_plus_plus,",
    ]

    sig = (itemty, f32arrty, f32arrty)
    disp = dpex.kernel(sig, debug=debug_option)(foo)
    kcres = disp.get_compile_result(types.void(itemty, f32arrty, f32arrty))
    kernel_ir = kcres.library.get_llvm_str()

    for tag in ir_tags:
        assert make_check(kernel_ir, tag)
