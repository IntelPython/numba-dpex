# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
from types import FunctionType

from numba.core import compiler, ir
from numba.core import types as numba_types
from numba.core import utils
from numba.core.caching import NullCache
from numba.core.compiler_lock import global_compiler_lock

from numba_dpex import compiler as dpex_compiler
from numba_dpex import config, spirv_generator
from numba_dpex.core.caching import SpirvKernelCache
from numba_dpex.core.descriptor import dpex_target
from numba_dpex.core.exceptions import (
    KernelHasReturnValueError,
    UncompiledKernelError,
    UnreachableError,
)

from .kernel_base import KernelInterface


class SpirvKernel(KernelInterface):
    def __init__(self, pyfunc, pyfunc_name) -> None:
        self._llvm_ir_str = None
        self._device_driver_ir_module = None
        self._module_name = None
        self._pyfunc_name = pyfunc_name
        self._pyfunc = pyfunc
        if isinstance(pyfunc, FunctionType):
            self._func_ty = FunctionType
        elif isinstance(pyfunc, ir.FunctionIR):
            self._func_ty = ir.FunctionIR
        else:
            raise UnreachableError()

        self._cache = NullCache

    def enable_caching(self):
        self._cache = SpirvKernelCache(self._pyfunc)

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

        # TODO: open an reproducible ticket
        # cres.codegen = SpirvCodeLibrary
        # why codegen doesn't exist in cres?

        # Linking depending libraries
        library = cres.library
        library.finalize()

        return cres

    @property
    def llvm_module(self):
        """The LLVM IR Module corresponding to the Kernel instance."""
        if self._llvm_ir_str:
            return self._llvm_ir_str
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

    def get_device_driver_ir_module(self):
        return self._device_driver_ir_module

    def set_device_driver_ir_module(self, ddir_module):
        self._device_driver_ir_module = ddir_module

    def compile(self, arg_types, debug, extra_compile_flags):
        """_summary_

        Args:
            arg_types (_type_): _description_
            debug (_type_): _description_
            extra_compile_flags (_type_): _description_
        """

        logging.debug("compiling SpirvKernel with arg types", arg_types)

        print("-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().1")
        sig = utils.pysignature(self._pyfunc)
        data = self._cache.load_overload(sig, dpex_target.target_context)
        print(
            "-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().2",
            "ddir_module =",
            data[1] if data is not None else data,
        )
        if data is not None:
            print(
                "-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().3"
            )
            self._device_driver_ir_module = data[0]
            self._module_name = data[1]
        else:
            print(
                "-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().4"
            )
            cres = self._compile(
                pyfunc=self._pyfunc,
                args=arg_types,
                debug=debug,
                extra_compile_flags=extra_compile_flags,
            )

            self._target_context = cres.target_context

            func = cres.library.get_function(cres.fndesc.llvm_func_name)
            ocl_kernel = cres.target_context.prepare_ocl_kernel(
                func, cres.signature.args
            )

            self._llvm_ir_str = ocl_kernel.module.__str__()
            print(
                "-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().5",
                "self._llvm_ir_str =",
                self._llvm_ir_str[0:10],
            )
            self._module_name = ocl_kernel.name
            print(
                "-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().6",
                "self._module_name =",
                self._module_name[0:10],
            )

            # FIXME: There is no need to serialize the bitcode. It can be passed to
            # llvm-spirv directly via stdin.

            # FIXME: There is no need for spirv-dis. We cause use --to-text
            # (or --spirv-text) to convert SPIRV to text
            ddir_module = spirv_generator.llvm_to_spirv(
                self._target_context,
                self._llvm_ir_str,
                ocl_kernel.module.as_bitcode(),
            )

            print(
                "-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().7",
                "ddir_module =",
                ddir_module[0:10],
            )
            # cache self._device_driver_ir_module
            print(
                "-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().8"
            )
            self._cache.save_overload(
                sig, (ddir_module, self._module_name), self._target_context
            )
            print(
                "-----> numba_dpex.core.kernel_interface.SpirvKernel.compile().9"
            )

            self._device_driver_ir_module = ddir_module