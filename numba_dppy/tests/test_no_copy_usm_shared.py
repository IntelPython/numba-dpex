import pytest
import numpy as np
from numba import njit, prange

from numba.core import compiler, cpu

import dpctl.dptensor.numpy_usm_shared as usmarray
import numba_dppy.numpy_usm_shared as nus

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
    flags.set('auto_parallel', cpu.ParallelOptions(True))
    flags.unset("nrt")

    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    args = typingctx.resolve_argument_type(a)

    cres = compiler.compile_extra(
		typingctx=typingctx,
		targetctx=targetctx,
		func=fn,
		args=tuple([args]),
		return_type=args,
		flags=flags,
		locals={},
		pipeline_class=DPPYCompiler,)


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
		pipeline_class=DPPYCompiler,)


    assert "DPCTLQueue_Memcpy" in cres.library.get_llvm_str()
