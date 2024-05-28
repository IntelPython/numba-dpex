# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a new numba dispatcher class and a compiler class to compile and
call numba_dpex.kernel decorated function.
"""
import hashlib
from collections import namedtuple
from contextlib import ExitStack
from typing import List, Tuple

import numba.core.event as ev
from llvmlite.binding.value import ValueRef
from numba.core import errors, sigutils, types
from numba.core.compiler import CompileResult, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher, _FunctionCompiler
from numba.core.funcdesc import PythonFunctionDescriptor
from numba.core.target_extension import (
    dispatcher_registry,
    target_override,
    target_registry,
)
from numba.core.types import Array as NpArrayType
from numba.core.types import void
from numba.core.typing.typeof import Purpose, typeof

from numba_dpex.core import config
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.core.exceptions import (
    ExecutionQueueInferenceError,
    InvalidKernelSpecializationError,
    KernelHasReturnValueError,
    UnsupportedKernelArgumentError,
)
from numba_dpex.core.pipelines import kernel_compiler
from numba_dpex.core.types import USMNdArray
from numba_dpex.core.utils import call_kernel_builder as kl
from numba_dpex.kernel_api_impl.spirv import spirv_generator
from numba_dpex.kernel_api_impl.spirv.codegen import SPIRVCodeLibrary
from numba_dpex.kernel_api_impl.spirv.target import (
    CompilationMode,
    SPIRVTargetContext,
)

from .target import SPIRV_TARGET_NAME

_SPIRVKernelCompileResult = namedtuple(
    "_KernelCompileResult", CompileResult._fields + ("kernel_device_ir_module",)
)


class _SPIRVKernelCompiler(_FunctionCompiler):
    """A special compiler class used to compile numba_dpex.kernel decorated
    functions.
    """

    def check_queue_equivalence_of_args(
        self, py_func_name: str, args: List[types.Type]
    ):
        """Evaluates if all USMNdArray arguments passed to a kernel function
        has the same DpctlSyclQueue type.

        Args:
            py_func_name (str): Name of the kernel that is being evaluated
            args (types.Type, ...]): List of numba inferred types for each
            argument passed to the kernel

        Raises:
            ExecutionQueueInferenceError: If all USMNdArray were not allocated
            on the same dpctl.SyclQueue
            ExecutionQueueInferenceError: If there were not USMNdArray
            arguments passed to the kernel.
        """
        common_queue = None

        for arg in args:
            if isinstance(arg, USMNdArray):
                if common_queue is None:
                    common_queue = arg.queue
                elif common_queue != arg.queue:
                    raise ExecutionQueueInferenceError(
                        kernel_name=py_func_name, usmarray_argnum_list=[]
                    )

        if common_queue is None:
            raise ExecutionQueueInferenceError(
                kernel_name=py_func_name, usmarray_argnum_list=None
            )

    def check_sig_types(self, py_func_name: str, argtypes, return_type):
        """Checks signature type and raises error if wrong signature was
        provided.

        Args:
            sig: The signature on which the kernel is to be specialized.

        Raises:
            KernelHasReturnValueError: non void return type.
            InvalidKernelSpecializationError: unsupported arguments where
                provided.
        """

        if return_type is None:
            return_type = void
        sig = return_type(*argtypes)

        # Check if signature has a non-void return type
        if return_type and return_type != void:
            raise KernelHasReturnValueError(
                kernel_name=None,
                return_type=return_type,
                sig=sig,
            )

        # USMNdarray check
        usmarray_argnums = []
        usmndarray_argtypes = []
        unsupported_argnum_list = []

        for i, argtype in enumerate(argtypes):
            # FIXME: Add checks for other types of unsupported kernel args, e.g.
            # complex.

            # Check if a non-USMNdArray Array type is passed to the kernel
            if isinstance(argtype, NpArrayType) and not isinstance(
                argtype, USMNdArray
            ):
                unsupported_argnum_list.append(i)
            elif isinstance(argtype, USMNdArray):
                usmarray_argnums.append(i)
                usmndarray_argtypes.append(argtype)

        if unsupported_argnum_list:
            raise InvalidKernelSpecializationError(
                kernel_name=py_func_name,
                invalid_sig=sig,
                unsupported_argnum_list=unsupported_argnum_list,
            )

    def check_arguments(self, py_func_name: str, args: List[types.Type]):
        """Checks arguments and queue of the input arguments.

        Raises:
            KernelHasReturnValueError: non void return type.
            InvalidKernelSpecializationError: unsupported arguments where
                provided.
            ExecutionQueueInferenceError: If all USMNdArray were not allocated
                on the same dpctl.SyclQueue
            ExecutionQueueInferenceError: If there were not USMNdArray
                arguments passed to the kernel.
        """
        self.check_sig_types(py_func_name, args, None)
        self.check_queue_equivalence_of_args(py_func_name, args)

    def _compile_to_spirv(
        self,
        kernel_library: SPIRVCodeLibrary,
        kernel_fndesc: PythonFunctionDescriptor,
        kernel_targetctx: SPIRVTargetContext,
    ):
        kernel_func: ValueRef = kernel_library.get_function(
            kernel_fndesc.llvm_func_name
        )

        # Create a spir_kernel wrapper function
        kernel_fn = kernel_targetctx.prepare_spir_kernel(
            kernel_func, kernel_fndesc.argtypes
        )
        # Get the compiler flags that were passed through the target descriptor
        flags = Flags()
        self.targetdescr.options.parse_as_flags(flags, self.targetoptions)

        # If the inline_threshold option was set then set the property in the
        # kernel_library to force inlining ``overload`` calls into a kernel.
        inline_threshold = flags.inline_threshold  # pylint: disable=E1101
        kernel_library.inline_threshold = inline_threshold

        # Call finalize on the LLVM module. Finalization will result in
        # all linking libraries getting linked together and final optimization
        # including inlining of functions if an inlining level is specified.
        kernel_library.finalize()

        if config.DUMP_KERNEL_LLVM:
            self._dump_kernel(kernel_fndesc, kernel_library)
        # Compiled the LLVM IR to SPIR-V
        kernel_spirv_module = spirv_generator.llvm_to_spirv(
            kernel_targetctx,
            kernel_library.final_module,
            kernel_library.final_module.as_bitcode(),
        )
        return kl.SPIRVKernelModule(
            kernel_name=kernel_fn.name, kernel_bitcode=kernel_spirv_module
        )

    def compile(self, args, return_type) -> _SPIRVKernelCompileResult:
        status, kcres = self._compile_cached(args, return_type)
        if status:
            return kcres

        raise kcres

    def _compile_cached(
        self, args, return_type: types.Type
    ) -> Tuple[bool, _SPIRVKernelCompileResult]:
        """Compiles the kernel function to bitcode and generates a host-callable
        wrapper to submit the kernel to a SYCL queue.

        The LLVM IR generated for the kernel function is available in the
        CompileResult objected returned by
        numba_dpex.core.pipeline.kernel_compiler.KernelCompiler.

        Once the kernel decorated function is compiled down to LLVM IR, the
        following steps are performed:

            a) compile the IR into SPIR-V kernel
            b) generate a host callable wrapper function that will create a
               sycl::kernel_bundle from the SPIR-V and then submits the
               kernel_bundle to a sycl::queue
            c) create a cpython_wrapper_function for the host callable wrapper
               function.
            d) create a cfunc_wrapper_function to make the host callable wrapper
               function callable inside another JIT-compiled function.

        Args:
            args (tuple(types.Type)): A tuple of numba.core.Type instances each
            representing the numba-inferred type of a kernel argument.

            return_type (types.Type): The numba-inferred type of the returned
            value from the kernel. Should always be types.NoneType.

        Returns:
            CompileResult: A CompileResult object storing the LLVM library for
            the host-callable wrapper function.
        """
        key = tuple(args), return_type
        try:
            return False, self._failed_cache[key]
        except KeyError:
            pass

        try:
            with target_override(SPIRV_TARGET_NAME):
                cres: CompileResult = self._compile_core(args, return_type)

            if (
                self.targetoptions["_compilation_mode"]
                == CompilationMode.KERNEL
            ):
                kernel_device_ir_module: kl.SPIRVKernelModule = (
                    self._compile_to_spirv(
                        cres.library, cres.fndesc, cres.target_context
                    )
                )
            else:
                kernel_device_ir_module = None

            kcres_attrs = []

            for cres_field in cres._fields:
                cres_attr = getattr(cres, cres_field)
                if cres_field == "entry_point":
                    if cres_attr is not None:
                        raise AssertionError(
                            "Compiled kernel and device_func should be "
                            "compiled with compile_cfunc option turned off"
                        )
                    cres_attr = cres.fndesc.qualname
                kcres_attrs.append(cres_attr)

            kcres_attrs.append(kernel_device_ir_module)

        except errors.TypingError as err:
            self._failed_cache[key] = err
            return False, err

        return True, _SPIRVKernelCompileResult(*kcres_attrs)

    def _dump_kernel(self, fndesc, library):
        """Dump kernel into file."""
        name = fndesc.llvm_func_name
        if len(name) > 200:
            sha256 = hashlib.sha256(name.encode("utf-8")).hexdigest()
            name = name[:150] + "_" + sha256

        with open(
            name + ".ll",
            "w",
            encoding="UTF-8",
        ) as fptr:
            fptr.write(str(library.final_module))


class SPIRVKernelDispatcher(Dispatcher):
    """Dispatcher class designed to compile kernel decorated functions. The
    dispatcher inherits the Numba Dispatcher class, but has a different
    compilation strategy. Instead of compiling a kernel decorated function to
    an executable binary, the dispatcher compiles it to SPIR-V and then caches
    that SPIR-V bitcode.

    """

    targetdescr = dpex_kernel_target
    _fold_args = False

    def __init__(
        self,
        pyfunc,
        local_vars_to_numba_types=None,
        targetoptions=None,
        pipeline_class=kernel_compiler.KernelCompiler,
    ):
        if targetoptions is None:
            targetoptions = {}

        if local_vars_to_numba_types is None:
            local_vars_to_numba_types = {}

        targetoptions["nopython"] = True
        targetoptions["experimental"] = True

        self._kernel_name = pyfunc.__name__

        super().__init__(
            py_func=pyfunc,
            locals=local_vars_to_numba_types,
            targetoptions=targetoptions,
            pipeline_class=pipeline_class,
        )
        self._compiler = _SPIRVKernelCompiler(
            pyfunc,
            self.targetdescr,
            targetoptions,
            local_vars_to_numba_types,
            pipeline_class,
        )

    def typeof_pyval(self, val):
        """
        Resolve the Numba type of Python value *val*.
        This is called from numba._dispatcher as a fallback if the native code
        cannot decide the type.
        """
        # Not going through the resolve_argument_type() indirection
        # can save a couple µs.
        try:
            typ = typeof(val, Purpose.argument)
            if isinstance(typ, types.Array) and not isinstance(typ, USMNdArray):
                raise UnsupportedKernelArgumentError(
                    type=str(type(val)), value=val
                )
        except ValueError:
            typ = types.pyobject
        else:
            if typ is None:
                typ = types.pyobject
        self._types_active_call.append(typ)
        return typ

    def add_overload(self, cres):
        args = tuple(cres.signature.args)
        self.overloads[args] = cres

    def compile(self, sig) -> any:
        disp = self._get_dispatcher_for_current_target()
        if disp is not self:
            return disp.compile(sig)

        with ExitStack() as scope:
            cres = None

            def cb_compiler(dur):
                if cres is not None:
                    self._callback_add_compiler_timer(dur, cres)

            def cb_llvm(dur):
                if cres is not None:
                    self._callback_add_llvm_timer(dur, cres)

            scope.enter_context(
                ev.install_timer("numba:compiler_lock", cb_compiler)
            )
            scope.enter_context(ev.install_timer("numba:llvm_lock", cb_llvm))
            scope.enter_context(global_compiler_lock)

            if not self._can_compile:
                raise RuntimeError("compilation disabled")
            # Use counter to track recursion compilation depth
            with self._compiling_counter:
                args, return_type = sigutils.normalize_signature(sig)

                if (
                    self.targetoptions["_compilation_mode"]
                    == CompilationMode.KERNEL
                ):
                    # Compute follows data based queue equivalence is only
                    # evaluated for kernel functions whose arguments are
                    # supposed to be arrays. For device_func decorated
                    # functions, the arguments can be scalar and we skip queue
                    # equivalence check.
                    try:
                        self._compiler.check_arguments(self._kernel_name, args)
                    except ExecutionQueueInferenceError as eqie:
                        raise eqie

                # Don't recompile if signature already exists
                existing = self.overloads.get(tuple(args))
                if existing is not None:
                    return existing.entry_point

                # TODO: Enable caching
                # Add code to enable on disk caching of a binary spirv kernel.
                # Refer: https://github.com/IntelPython/numba-dpex/issues/1197
                self._cache_misses[sig] += 1
                with ev.trigger_event(
                    "numba_dpex:compile",
                    data={
                        "dispatcher": self,
                        "args": args,
                        "return_type": return_type,
                    },
                ):
                    try:
                        compiler: _SPIRVKernelCompiler = self._compiler
                        kcres: _SPIRVKernelCompileResult = compiler.compile(
                            args, return_type
                        )
                        if (
                            self.targetoptions["_compilation_mode"]
                            == CompilationMode.KERNEL
                            and kcres.signature.return_type is not None
                            and kcres.signature.return_type != types.void
                        ):
                            raise KernelHasReturnValueError(
                                kernel_name=self.py_func.__name__,
                                return_type=kcres.signature.return_type,
                            )
                    except errors.ForceLiteralArg as err:

                        def folded(args, kws):
                            return self._compiler.fold_argument_types(
                                args, kws
                            )[1]

                        raise err.bind_fold_arguments(folded)
                    self.add_overload(kcres)

                    kcres.target_context.insert_user_function(
                        kcres.entry_point, kcres.fndesc, [kcres.library]
                    )

                # TODO: enable caching of kernel_module
                # https://github.com/IntelPython/numba-dpex/issues/1197

                return kcres.entry_point

    def __getitem__(self, args):
        """Square-bracket notation for configuring launch arguments is not
        supported.
        """

        raise NotImplementedError

    def __call__(self, *args, **kw_args):
        """Functor to launch a kernel."""

        raise NotImplementedError


_dpex_target = target_registry[SPIRV_TARGET_NAME]
dispatcher_registry[_dpex_target] = SPIRVKernelDispatcher
