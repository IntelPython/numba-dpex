# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a new numba dispatcher class and a compiler class to compile and
call numba_dpex.kernel decorated function.
"""
from collections import namedtuple
from contextlib import ExitStack
from typing import Tuple

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
from numba.core.types import void
from numba.core.typing.typeof import Purpose, typeof

from numba_dpex import config, numba_sem_version, spirv_generator
from numba_dpex.core.codegen import SPIRVCodeLibrary
from numba_dpex.core.exceptions import (
    ExecutionQueueInferenceError,
    KernelHasReturnValueError,
    UnsupportedKernelArgumentError,
)
from numba_dpex.core.pipelines import kernel_compiler
from numba_dpex.core.targets.kernel_target import (
    CompilationMode,
    DpexKernelTargetContext,
)
from numba_dpex.core.types import DpnpNdArray
from numba_dpex.core.utils import kernel_launcher as kl

from .target import DPEX_KERNEL_EXP_TARGET_NAME, dpex_exp_kernel_target

_KernelCompileResult = namedtuple(
    "_KernelCompileResult", CompileResult._fields + ("kernel_device_ir_module",)
)


class _KernelCompiler(_FunctionCompiler):
    """A special compiler class used to compile numba_dpex.kernel decorated
    functions.
    """

    def check_queue_equivalence_of_args(
        self, py_func_name: str, args: [types.Type, ...]
    ):
        """Evaluates if all DpnpNdArray arguments passed to a kernel function
        has the same DpctlSyclQueue type.

        Args:
            py_func_name (str): Name of the kernel that is being evaluated
            args (types.Type, ...]): List of numba inferred types for each
            argument passed to the kernel

        Raises:
            ExecutionQueueInferenceError: If all DpnpNdArray were not allocated
            on the same dpctl.SyclQueue
            ExecutionQueueInferenceError: If there were not DpnpNdArray
            arguments passed to the kernel.
        """
        common_queue = None

        for arg in args:
            if isinstance(arg, DpnpNdArray):
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

    def _compile_to_spirv(
        self,
        kernel_library: SPIRVCodeLibrary,
        kernel_fndesc: PythonFunctionDescriptor,
        kernel_targetctx: DpexKernelTargetContext,
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
        # Compiled the LLVM IR to SPIR-V
        kernel_spirv_module = spirv_generator.llvm_to_spirv(
            kernel_targetctx,
            kernel_library.final_module,
            kernel_library.final_module.as_bitcode(),
        )
        return kl.SPIRVKernelModule(
            kernel_name=kernel_fn.name, kernel_bitcode=kernel_spirv_module
        )

    def compile(self, args, return_type) -> _KernelCompileResult:
        status, kcres = self._compile_cached(args, return_type)
        if status:
            return kcres

        raise kcres

    def _compile_cached(
        self, args, return_type: types.Type
    ) -> Tuple[bool, _KernelCompileResult]:
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
            with target_override(DPEX_KERNEL_EXP_TARGET_NAME):
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

            if config.DUMP_KERNEL_LLVM:
                with open(
                    cres.fndesc.llvm_func_name + ".ll",
                    "w",
                    encoding="UTF-8",
                ) as f:
                    f.write(cres.library.final_module)

        except errors.TypingError as e:
            self._failed_cache[key] = e
            return False, e

        return True, _KernelCompileResult(*kcres_attrs)


class KernelDispatcher(Dispatcher):
    """Dispatcher class designed to compile kernel decorated functions. The
    dispatcher inherits the Numba Dispatcher class, but has a different
    compilation strategy. Instead of compiling a kernel decorated function to
    an executable binary, the dispatcher compiles it to SPIR-V and then caches
    that SPIR-V bitcode.

    """

    targetdescr = dpex_exp_kernel_target
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

        if numba_sem_version < (0, 59, 0):
            super().__init__(
                py_func=pyfunc,
                locals=local_vars_to_numba_types,
                impl_kind="direct",
                targetoptions=targetoptions,
                pipeline_class=pipeline_class,
            )
        else:
            super().__init__(
                py_func=pyfunc,
                locals=local_vars_to_numba_types,
                targetoptions=targetoptions,
                pipeline_class=pipeline_class,
            )
        self._compiler = _KernelCompiler(
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
            tp = typeof(val, Purpose.argument)
            if isinstance(tp, types.Array) and not isinstance(tp, DpnpNdArray):
                raise UnsupportedKernelArgumentError(
                    type=str(type(val)), value=val
                )
        except ValueError:
            tp = types.pyobject
        else:
            if tp is None:
                tp = types.pyobject
        self._types_active_call.append(tp)
        return tp

    def add_overload(self, cres):
        args = tuple(cres.signature.args)
        self.overloads[args] = cres

    def get_overload_kcres(self, sig) -> _KernelCompileResult:
        """
        Return the compiled function for the given signature.
        """
        args, _ = sigutils.normalize_signature(sig)
        return self.overloads[tuple(args)]

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
                        self._compiler.check_queue_equivalence_of_args(
                            self._kernel_name, args
                        )
                    except ExecutionQueueInferenceError as eqie:
                        raise eqie

                    # A function being compiled in the KERNEL compilation mode
                    # cannot have a non-void return value
                    if return_type and return_type != void:
                        raise KernelHasReturnValueError(
                            kernel_name=None, return_type=return_type, sig=sig
                        )

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
                        compiler: _KernelCompiler = self._compiler
                        kcres: _KernelCompileResult = compiler.compile(
                            args, return_type
                        )
                    except errors.ForceLiteralArg as e:

                        def folded(args, kws):
                            return self._compiler.fold_argument_types(
                                args, kws
                            )[1]

                        raise e.bind_fold_arguments(folded)
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


_dpex_target = target_registry[DPEX_KERNEL_EXP_TARGET_NAME]
dispatcher_registry[_dpex_target] = KernelDispatcher
