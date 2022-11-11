# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor.numpy_usm_shared as usmarray
import numpy as np
import pytest
from numba import prange
from numba.core import compiler, cpu
from numba.core.registry import cpu_target

from numba_dpex.compiler import Compiler
from numba_dpex.tests._helper import skip_no_opencl_gpu


def fn(a):
    for i in prange(a.size):
        a[i] += 1
    return a


@skip_no_opencl_gpu
def test_no_copy_usm_shared(capfd):
    a = usmarray.ones(10, dtype=np.int64)
    b = np.ones(10, dtype=np.int64)
    # f = njit(fn)

    flags = compiler.Flags()
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.nrt = False
    flags.auto_parallel = cpu.ParallelOptions(True)

    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    args = typingctx.resolve_argument_type(a)

    device = dpctl.SyclDevice("opencl:gpu:0")

    with dpctl.device_context(device):
        cres = compiler.compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=fn,
            args=tuple([args]),
            return_type=args,
            flags=flags,
            locals={},
            pipeline_class=Compiler,
        )

        assert "DPCTLQueue_Memcpy" not in cres.library.get_llvm_str()

        args = typingctx.resolve_argument_type(b)
        cres = compiler.compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=fn,
            args=tuple([args]),
            return_type=args,
            flags=flags,
            locals={},
            pipeline_class=Compiler,
        )

        assert "DPCTLQueue_Memcpy" in cres.library.get_llvm_str()
