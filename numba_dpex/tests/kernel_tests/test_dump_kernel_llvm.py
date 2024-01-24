#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os

import numba_dpex as dpex
from numba_dpex import float32, usm_ndarray
from numba_dpex.core import config
from numba_dpex.core.descriptor import dpex_kernel_target

f32arrty = usm_ndarray(ndim=1, dtype=float32, layout="C")


def _get_kernel_llvm(fn, sig, debug=False):
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
    return kernel.module_name, kernel.llvm_module


def test_dump_file_on_dump_kernel_llvm_flag_on():
    """
    Tests functionality of DUMP_KERNEL_LLVM config variable.

    Check llvm source is dumped into a .ll file in current directory
    and compare with llvm source stored in SprivKernel.
    """

    def data_parallel_sum(var_a, var_b, var_c):
        i = dpex.get_global_id(0)
        var_c[i] = var_a[i] + var_b[i]

    sig = (f32arrty, f32arrty, f32arrty)

    config.DUMP_KERNEL_LLVM = True

    llvm_module_name, llvm_module_str = _get_kernel_llvm(data_parallel_sum, sig)

    dump_file_name = (
        "llvm_kernel_"
        + hashlib.sha256(llvm_module_name.encode()).hexdigest()
        + ".ll"
    )

    with open(dump_file_name, "r") as f:
        llvm_dump = f.read()

    assert llvm_module_str == llvm_dump

    os.remove(dump_file_name)


def test_no_dump_file_on_dump_kernel_llvm_flag_off():
    """
    Test functionality of DUMP_KERNEL_LLVM config variable.
    Check llvm source is not dumped in .ll file in current directory.
    """

    def data_parallel_sum(var_a, var_b, var_c):
        i = dpex.get_global_id(0)
        var_c[i] = var_a[i] + var_b[i]

    sig = (f32arrty, f32arrty, f32arrty)

    config.DUMP_KERNEL_LLVM = False

    llvm_module_name, llvm_module_str = _get_kernel_llvm(data_parallel_sum, sig)

    dump_file_name = (
        "llvm_kernel_"
        + hashlib.sha256(llvm_module_name.encode()).hexdigest()
        + ".ll"
    )

    assert not os.path.isfile(dump_file_name)
