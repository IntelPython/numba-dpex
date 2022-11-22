# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
from types import FunctionType

from numba.core import ir

from numba_dpex import spirv_generator
from numba_dpex.core import _compile_helper
from numba_dpex.core.exceptions import UncompiledKernelError, UnreachableError

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

        cres = _compile_helper.compile_with_dpex(
            self._func,
            self._pyfunc_name,
            args=arg_types,
            return_type=None,
            debug=debug,
            is_kernel=True,
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
