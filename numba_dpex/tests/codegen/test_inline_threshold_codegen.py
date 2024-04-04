# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
from numba.core import types

import numba_dpex as dpex
from numba_dpex import DpctlSyclQueue, DpnpNdArray, int64
from numba_dpex.core.types.kernel_api.index_space_ids import ItemType
from numba_dpex.kernel_api import Item


def kernel_func(item: Item, a, b, c):
    i = item.get_id(0)
    c[i] = a[i] + b[i]


def test_codegen_with_max_inline_threshold():
    """Tests if the inline_threshold option leads to a fully inlined kernel
    function generation.

    By default, numba_dpex compiles a function passed to the `kernel` decorator
    into a `spir_func` LLVM function. Then before lowering to device IR, the
    DpexTargetContext creates a "wrapper" function that has the "spir_kernel"
    calling convention. It is done so that we can use the same target context
    and pipeline to compile both host callable "kernels" and device-only
    "device_func" functions.

    Unless the inline_threshold is set to >0, the `spir_func` function is not
    inlined into the wrapper function. The test checks if the `spir_func`
    function is fully inlined into the wrapper. The test is rather rudimentary
    and only checks the count of function in the generated module.
    With inlining, the count should be one and without inlining it will be two.
    """

    queue_ty = DpctlSyclQueue(dpctl.SyclQueue())
    i64arr_ty = DpnpNdArray(ndim=1, dtype=int64, layout="C", queue=queue_ty)
    kernel_sig = types.void(ItemType(1), i64arr_ty, i64arr_ty, i64arr_ty)

    disp = dpex.kernel(inline_threshold=1)(kernel_func)
    disp.compile(kernel_sig)
    kcres = disp.overloads[kernel_sig.args]
    llvm_ir_mod = kcres.library._final_module

    count_of_non_declaration_type_functions = 0

    for f in llvm_ir_mod.functions:
        if not f.is_declaration:
            count_of_non_declaration_type_functions += 1

    assert count_of_non_declaration_type_functions == 1


def test_codegen_without_max_inline_threshold():
    """See docstring of :func:`test_codegen_with_max_inline_threshold`."""

    queue_ty = DpctlSyclQueue(dpctl.SyclQueue())
    i64arr_ty = DpnpNdArray(ndim=1, dtype=int64, layout="C", queue=queue_ty)
    kernel_sig = types.void(ItemType(1), i64arr_ty, i64arr_ty, i64arr_ty)

    disp = dpex.kernel(inline_threshold=0)(kernel_func)
    disp.compile(kernel_sig)
    kcres = disp.overloads[kernel_sig.args]
    llvm_ir_mod = kcres.library._final_module

    count_of_non_declaration_type_functions = 0

    for f in llvm_ir_mod.functions:
        if not f.is_declaration:
            count_of_non_declaration_type_functions += 1

    assert count_of_non_declaration_type_functions == 3
