# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import re

import dpctl
from numba.core import types

import numba_dpex as dpex
from numba_dpex import DpctlSyclQueue, DpnpNdArray, int64
from numba_dpex.kernel_api.flag_enum import FlagEnum


def test_compilation_as_literal_constant():
    """Tests if FlagEnum objects are treaded as scalar constants inside
    numba-dpex generated code.

    The test case compiles the kernel `pass_flags_to_func` that includes a
    call to the device_func `bitwise_or_flags`. The `bitwise_or_flags` function
    is passed two FlagEnum arguments. The test case evaluates the generated
    LLVM IR for `pass_flags_to_func` to see if the call to `bitwise_or_flags`
    has the scalar arguments `i64 1` and `i64 2`.
    """

    class PseudoFlags(FlagEnum):
        FLAG1 = 1
        FLAG2 = 2

    @dpex.device_func
    def bitwise_or_flags(flag1, flag2):
        return flag1 | flag2

    def pass_flags_to_func(a):
        f1 = PseudoFlags.FLAG1
        f2 = PseudoFlags.FLAG2
        a[0] = bitwise_or_flags(f1, f2)

    queue_ty = DpctlSyclQueue(dpctl.SyclQueue())
    i64arr_ty = DpnpNdArray(ndim=1, dtype=int64, layout="C", queue=queue_ty)
    kernel_sig = types.void(i64arr_ty)

    disp = dpex.kernel(inline_threshold=0)(pass_flags_to_func)
    disp.compile(kernel_sig)
    kcres = disp.overloads[kernel_sig.args]
    llvm_ir_mod = kcres.library._final_module.__str__()

    pattern = re.compile(
        r"call spir_func i32 @\_Z.*bitwise\_or"
        r"\_flags.*\(i64\*\s(\w+)?\s*%.*, i64 1, i64 2\)"
    )

    assert re.search(pattern, llvm_ir_mod) is not None
