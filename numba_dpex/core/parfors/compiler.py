# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import compiler, ir
from numba.core import types as numba_types
from numba.core.compiler_lock import global_compiler_lock

from numba_dpex.core import config
from numba_dpex.core.exceptions import (
    KernelHasReturnValueError,
    UnreachableError,
)
from numba_dpex.core.pipelines.kernel_compiler import KernelCompiler


@global_compiler_lock
def compile_numba_ir_with_dpex(
    pyfunc,
    pyfunc_name,
    args,
    return_type,
    target_context,
    typing_context,
    debug=False,
    is_kernel=True,
    extra_compile_flags=None,
):
    """
    Compiles a function using class:`numba_dpex.core.pipelines.KernelCompiler`
    and returns the compiled result.

    Args:
        args: The list of arguments passed to the kernel.
        debug (bool): Optional flag to turn on debug mode compilation.
        extra_compile_flags: Extra flags passed to the compiler.

    Returns:
        cres: Compiled result.

    Raises:
        KernelHasReturnValueError: If the compiled function returns a
            non-void value.
    """
    # First compilation will trigger the initialization of the backend.
    typingctx = typing_context
    targetctx = target_context

    flags = compiler.Flags()
    # Do not compile the function to a binary, just lower to LLVM
    flags.debuginfo = config.DEBUGINFO_DEFAULT
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True
    flags.nrt = False

    if debug:
        flags.debuginfo = debug

    # Run compilation pipeline
    if isinstance(pyfunc, ir.FunctionIR):
        cres = compiler.compile_ir(
            typingctx=typingctx,
            targetctx=targetctx,
            func_ir=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=KernelCompiler,
        )
    else:
        raise UnreachableError()

    if (
        is_kernel
        and cres.signature.return_type is not None
        and cres.signature.return_type != numba_types.void
    ):
        raise KernelHasReturnValueError(
            kernel_name=pyfunc_name,
            return_type=cres.signature.return_type,
        )
    # Linking depending libraries
    library = cres.library
    library.finalize()

    return cres
