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

import pytest
import numpy as np
from numba import njit, prange

from numba.core import compiler, cpu

import dpctl.dptensor.numpy_usm_shared as usmarray
import numba_dppy.numpy_usm_shared as nus
from numba_dppy.context import device_context

from numba_dppy.compiler import DPPYCompiler
from numba.core.registry import cpu_target


def fn(a):
    for i in prange(a.size):
        a[i] += 1
    return a


def test_no_copy_usm_shared(capfd):
    a = usmarray.ones(10, dtype=np.int64)
    b = np.ones(10, dtype=np.int64)
    f = njit(fn)

    flags = compiler.Flags()
    flags.set("no_compile")
    flags.set("no_cpython_wrapper")
    flags.set("auto_parallel", cpu.ParallelOptions(True))
    flags.unset("nrt")

    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    args = typingctx.resolve_argument_type(a)

    with device_context("opencl:gpu:0"):
        cres = compiler.compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=fn,
            args=tuple([args]),
            return_type=args,
            flags=flags,
            locals={},
            pipeline_class=DPPYCompiler,
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
            pipeline_class=DPPYCompiler,
        )

        assert "DPCTLQueue_Memcpy" in cres.library.get_llvm_str()
