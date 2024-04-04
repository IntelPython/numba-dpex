#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os

from numba.core import types

import numba_dpex as dpex
from numba_dpex import float32, usm_ndarray
from numba_dpex.core import config
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.core.types.kernel_api.index_space_ids import ItemType

f32arrty = usm_ndarray(ndim=1, dtype=float32, layout="C")
itemty = ItemType(ndim=1)


def _get_kernel_llvm():
    def data_parallel_sum(item, var_a, var_b, var_c):
        i = item.get_id(0)
        var_c[i] = var_a[i] + var_b[i]

    sig = (itemty, f32arrty, f32arrty, f32arrty)

    disp = dpex.kernel(sig)(data_parallel_sum)
    kcres = disp.get_compile_result(
        types.void(itemty, f32arrty, f32arrty, f32arrty)
    )
    llvm_module_str = kcres.library.get_llvm_str()
    name = kcres.fndesc.llvm_func_name
    if len(name) > 200:
        sha256 = hashlib.sha256(name.encode("utf-8")).hexdigest()
        name = name[:150] + "_" + sha256

    dump_file_name = name + ".ll"

    return dump_file_name, llvm_module_str


def test_dump_file_on_dump_kernel_llvm_flag_on():
    """
    Tests functionality of DUMP_KERNEL_LLVM config variable.

    Check llvm source is dumped into a .ll file in current directory
    and compare with llvm source stored in SprivKernel.
    """

    config.DUMP_KERNEL_LLVM = True
    dump_file_name, llvm_dumped_str = _get_kernel_llvm()
    with open(dump_file_name, "r") as f:
        ondisk_llvm_dump_str = f.read()
    assert llvm_dumped_str == ondisk_llvm_dump_str
    os.remove(dump_file_name)
    config.DUMP_KERNEL_LLVM = False


def test_no_dump_file_on_dump_kernel_llvm_flag_off():
    """
    Test functionality of DUMP_KERNEL_LLVM config variable.
    Check llvm source is not dumped in .ll file in current directory.
    """
    config.DUMP_KERNEL_LLVM = False
    dump_file_name, llvm_dumped_str = _get_kernel_llvm()
    assert not os.path.isfile(dump_file_name)
