# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
from types import FunctionType

from numba.core import ir, utils
from numba.core.caching import NullCache

from numba_dpex import spirv_generator
from numba_dpex.core import _compile_helper
from numba_dpex.core.caching import SpirvKernelCache
from numba_dpex.core.descriptor import dpex_target
from numba_dpex.core.exceptions import UncompiledKernelError, UnreachableError

from .kernel_base import KernelInterface


class SpirvKernel(KernelInterface):
    def __init__(self, func, pyfunc_name) -> None:
        self._llvm_module = None
        self._device_driver_ir_module = None
        self._module_name = None
        self._pyfunc_name = pyfunc_name
        self._func = func
        self._cache = NullCache
        if isinstance(func, FunctionType):
            self._func_ty = FunctionType
        elif isinstance(func, ir.FunctionIR):
            self._func_ty = ir.FunctionIR
        else:
            raise UnreachableError()

    def enable_caching(self):
        self._cache = SpirvKernelCache(self._func)

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

    def compile(
        self,
        arg_types,
        debug,
        extra_compile_flags,
        backend=None,
        device_type=None,
    ):
        """_summary_

        Args:
            arg_types (_type_): _description_
            debug (_type_): _description_
            extra_compile_flags (_type_): _description_
        """

        logging.debug("compiling SpirvKernel with arg types", arg_types)

        sig = utils.pysignature(self._func)

        # load the kernel from cache
        data = self._cache.load_overload(
            sig,
            dpex_target.target_context,
            backend=backend,
            device_type=device_type,
        )
        # if exists
        if data is not None:
            self._device_driver_ir_module, self._module_name = data
        else:  # otherwise, build from the scratch
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
            ocl_kernel = cres.target_context.prepare_ocl_kernel(
                func, cres.signature.args
            )

            self._llvm_ir_str = ocl_kernel.module.__str__()
            self._module_name = ocl_kernel.name

            # FIXME: There is no need to serialize the bitcode. It can be passed to
            # llvm-spirv directly via stdin.

            # FIXME: There is no need for spirv-dis. We cause use --to-text
            # (or --spirv-text) to convert SPIRV to text
            ddir_module = spirv_generator.llvm_to_spirv(
                self._target_context,
                self._llvm_ir_str,
                ocl_kernel.module.as_bitcode(),
            )

            # save into cache
            self._cache.save_overload(
                sig,
                (ddir_module, self._module_name),
                self._target_context,
                backend=backend,
                device_type=device_type,
            )

            self._device_driver_ir_module = ddir_module
