# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
from types import FunctionType

from numba.core import ir

from numba_dpex import spirv_generator
from numba_dpex._kernel_api_impl.spirv.target import SPIRVTargetContext
from numba_dpex.core import config
from numba_dpex.core.compiler import compile_with_dpex
from numba_dpex.core.exceptions import UncompiledKernelError, UnreachableError

from .kernel_base import KernelInterface


class SpirvKernel(KernelInterface):
    def __init__(self, func, func_name) -> None:
        """Represents a SPIR-V module compiled for a Python function.

        Args:
            func: The function to be compiled. Can be a Python function or a
            Numba IR object representing a function.
            func_name (str): Name of the function being compiled

        Raises:
            UnreachableError: An internal error indicating an unexpected code
            path was executed.
        """
        self._llvm_module = None
        self._device_driver_ir_module = None
        self._module_name = None
        self._pyfunc_name = func_name
        self._func = func
        if isinstance(func, FunctionType):
            self._func_ty = FunctionType
        elif isinstance(func, ir.FunctionIR):
            self._func_ty = ir.FunctionIR
        else:
            raise UnreachableError()
        self._target_context = None

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

    @property
    def target_context(self):
        """Returns the target context that was used to compile the kernel.

        Raises:
            UncompiledKernelError: If the kernel was not yet compiled.

        Returns:
            target context used to compile the kernel
        """
        if self._target_context:
            return self._target_context
        else:
            raise UncompiledKernelError(self._pyfunc_name)

    @property
    def typing_context(self):
        """Returns the typing context that was used to compile the kernel.

        Raises:
            UncompiledKernelError: If the kernel was not yet compiled.

        Returns:
            typing context used to compile the kernel
        """
        if self._typing_context:
            return self._typing_context
        else:
            raise UncompiledKernelError(self._pyfunc_name)

    def compile(
        self,
        target_ctx,
        typing_ctx,
        args,
        debug,
        compile_flags,
    ):
        """Compiles a kernel using numba_dpex.core.compiler.Compiler.

        Args:
            args (_type_): _description_
            debug (_type_): _description_
            compile_flags (_type_): _description_
        """

        logging.debug("compiling SpirvKernel with arg types", args)

        self._target_context = target_ctx
        self._typing_context = typing_ctx

        cres = compile_with_dpex(
            self._func,
            self._pyfunc_name,
            args=args,
            return_type=None,
            debug=debug,
            is_kernel=True,
            typing_context=typing_ctx,
            target_context=target_ctx,
            extra_compile_flags=compile_flags,
        )

        func = cres.library.get_function(cres.fndesc.llvm_func_name)
        kernel_targetctx: SPIRVTargetContext = cres.target_context
        kernel = kernel_targetctx.prepare_spir_kernel(func, cres.signature.args)

        # XXX: Setting the inline_threshold in the following way is a temporary
        # workaround till the JitKernel dispatcher is replaced by
        # experimental.dispatcher.KernelDispatcher.
        if config.INLINE_THRESHOLD is not None:
            cres.library.inline_threshold = config.INLINE_THRESHOLD
        else:
            cres.library.inline_threshold = 0

        cres.library._optimize_final_module()
        self._llvm_module = kernel.module.__str__()
        self._module_name = kernel.name

        # Dump LLVM IR if DEBUG flag is set.
        if config.DUMP_KERNEL_LLVM:
            import hashlib

            # Embed hash of module name in the output file name
            # so that different kernels are written to separate files
            with open(
                "llvm_kernel_"
                + hashlib.sha256(self._module_name.encode()).hexdigest()
                + ".ll",
                "w",
            ) as f:
                f.write(self._llvm_module)

        # FIXME: There is no need to serialize the bitcode. It can be passed to
        # llvm-spirv directly via stdin.

        # FIXME: There is no need for spirv-dis. We cause use --to-text
        # (or --spirv-text) to convert SPIRV to text
        self._device_driver_ir_module = spirv_generator.llvm_to_spirv(
            self._target_context, self._llvm_module, kernel.module.as_bitcode()
        )
