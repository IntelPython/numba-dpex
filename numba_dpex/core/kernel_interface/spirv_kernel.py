# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
from types import FunctionType

from numba.core import compiler, ir
from numba.core import types as numba_types
from numba.core.compiler_lock import global_compiler_lock

from numba_dpex import compiler as dpex_compiler
from numba_dpex import config, spirv_generator
from numba_dpex.core.descriptor import dpex_target
from numba_dpex.core.exceptions import (
    KernelHasReturnValueError,
    UncompiledKernelError,
    UnreachableError,
)

from .kernel_base import KernelInterface


class SpirvKernel(KernelInterface):
    def __init__(self, func, pyfunc_name) -> None:
        self._llvm_module = None
        self._device_driver_ir_module = None
        self._module_name = None
        self._pyfunc_name = pyfunc_name
        self._func = func
        if isinstance(func, FunctionType):
            self._func_ty = FunctionType
        elif isinstance(func, ir.FunctionIR):
            self._func_ty = ir.FunctionIR
        else:
            raise UnreachableError()

    @global_compiler_lock
    def _compile(self, pyfunc, args, debug=None, extra_compile_flags=None):
        """
        Compiles the function using the dpex compiler pipeline and returns the
        compiled result.

        Args:
            pyfunc: The function to be compiled. Can be a Python function or a
            Numba IR object representing a function.
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
        typingctx = dpex_target.typing_context
        targetctx = dpex_target.target_context

        flags = compiler.Flags()
        # Do not compile the function to a binary, just lower to LLVM
        flags.debuginfo = config.DEBUGINFO_DEFAULT
        flags.no_compile = True
        flags.no_cpython_wrapper = True
        flags.nrt = False

        if debug is not None:
            flags.debuginfo = debug

        # Run compilation pipeline
        if isinstance(pyfunc, FunctionType):
            cres = compiler.compile_extra(
                typingctx=typingctx,
                targetctx=targetctx,
                func=pyfunc,
                args=args,
                return_type=None,
                flags=flags,
                locals={},
                pipeline_class=dpex_compiler.Compiler,
            )
        elif isinstance(pyfunc, ir.FunctionIR):
            cres = compiler.compile_ir(
                typingctx=typingctx,
                targetctx=targetctx,
                func_ir=pyfunc,
                args=args,
                return_type=None,
                flags=flags,
                locals={},
                pipeline_class=dpex_compiler.Compiler,
            )
        else:
            raise UnreachableError()

        if (
            cres.signature.return_type is not None
            and cres.signature.return_type != numba_types.void
        ):
            raise KernelHasReturnValueError(
                kernel_name=pyfunc.__name__,
                return_type=cres.signature.return_type,
            )
        # Linking depending libraries
        library = cres.library
        library.finalize()

        return cres

    @property
    def llvm_module(self):
        """The LLVM IR Module corresponding to the Kernel instance."""
        if self._llvm_module:
            return self._llvm_module
        else:
            raise UncompiledKernelError(self._pyfunc_name)

    @property
    def device_driver_ir_module(self):
        """The module in a device IR (such as SPIR-V or PTX) format."""
        if self._device_driver_ir_module:
            return self._device_driver_ir_module
        else:
            raise UncompiledKernelError(self._pyfunc_name)

    @property
    def pyfunc_name(self):
        """The Python function name corresponding to the kernel."""
        return self._pyfunc_name

    @property
    def module_name(self):
        """The name of the compiled LLVM module for the kernel."""
        if self._module_name:
            return self._module_name
        else:
            raise UncompiledKernelError(self._pyfunc_name)

    def compile(self, arg_types, debug, extra_compile_flags):
        """_summary_

        Args:
            arg_types (_type_): _description_
            debug (_type_): _description_
            extra_compile_flags (_type_): _description_
        """

        logging.debug("compiling SpirvKernel with arg types", arg_types)

        cres = self._compile(
            pyfunc=self._func,
            args=arg_types,
            debug=debug,
            extra_compile_flags=extra_compile_flags,
        )

        self._target_context = cres.target_context

        func = cres.library.get_function(cres.fndesc.llvm_func_name)
        kernel = cres.target_context.prepare_ocl_kernel(
            func, cres.signature.args
        )
        self._llvm_module = kernel.module.__str__()
        self._module_name = kernel.name

        # FIXME: There is no need to serialize the bitcode. It can be passed to
        # llvm-spirv directly via stdin.

        # FIXME: There is no need for spirv-dis. We cause use --to-text
        # (or --spirv-text) to convert SPIRV to text
        self._device_driver_ir_module = spirv_generator.llvm_to_spirv(
            self._target_context, self._llvm_module, kernel.module.as_bitcode()
        )
