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

import dpctl
import dpctl.tensor.numpy_usm_shared as usmarray
import numpy as np
import pytest
from numba import prange
from numba.core import compiler, cpu
from numba.core.registry import cpu_target

import numba_dppy as dppy
from numba_dppy.compiler import Compiler
from numba_dppy.tests._helper import skip_no_opencl_gpu


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
