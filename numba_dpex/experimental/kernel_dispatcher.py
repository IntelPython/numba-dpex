# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import Counter, OrderedDict, namedtuple
from contextlib import ExitStack

import numba.core.event as ev
from numba.core import errors, sigutils, types, utils
from numba.core.caching import NullCache
from numba.core.compiler import CompileResult
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher, _DispatcherBase, _FunctionCompiler
from numba.core.typing.typeof import Purpose, typeof

from numba_dpex import config, spirv_generator
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.core.exceptions import (
    InvalidKernelLaunchArgsError,
    UnsupportedKernelArgumentError,
)
from numba_dpex.core.kernel_interface.indexers import NdRange, Range
from numba_dpex.core.pipelines import kernel_compiler
from numba_dpex.core.types import DpnpNdArray

_KernelLauncherLowerResult = namedtuple(
    "_KernelLauncherLowerResult",
    ["sig", "fndesc", "library", "call_helper"],
)

_KernelModule = namedtuple("_KernelModule", ["kernel_name", "kernel_bitcode"])

_KernelCompileResult = namedtuple(
    "_KernelCompileResult",
    ["status", "cres_or_error", "kernel_module"],
)


class _KernelCompiler(_FunctionCompiler):
    def _compile_to_spirv(
        self, kernel_library, kernel_fndesc, kernel_targetctx
    ):
        kernel_func = kernel_library.get_function(kernel_fndesc.llvm_func_name)

        # Create a spir_kernel wrapper function
        kernel_fn = kernel_targetctx.prepare_spir_kernel(
            kernel_func, kernel_fndesc.argtypes
        )

        # makes sure that the spir_func is completely inlined into the
        # spir_kernel wrapper
        kernel_library._optimize_final_module()
        # Compiled the LLVM IR to SPIR-V
        kernel_spirv_module = spirv_generator.llvm_to_spirv(
            kernel_targetctx,
            kernel_library._final_module,
            kernel_library._final_module.as_bitcode(),
        )
        return _KernelModule(
            kernel_name=kernel_fn.name, kernel_bitcode=kernel_spirv_module
        )

    def compile(self, args, return_type):
        kcres = self._compile_cached(args, return_type)
        if kcres.status:
            return kcres
        else:
            raise kcres.cres_or_error

    def _compile_cached(
        self, kernel_args, return_type: types.Type
    ) -> _KernelCompileResult:
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
        key = tuple(kernel_args), return_type
        try:
            return _KernelCompileResult(False, self._failed_cache[key], None)
        except KeyError:
            pass

        try:
            kernel_cres: CompileResult = self._compile_core(
                kernel_args, return_type
            )

            kernel_library = kernel_cres.library
            kernel_fndesc = kernel_cres.fndesc
            kernel_targetctx = kernel_cres.target_context

            kernel_module = self._compile_to_spirv(
                kernel_library, kernel_fndesc, kernel_targetctx
            )

            if config.DUMP_KERNEL_LLVM:
                with open(
                    kernel_cres.fndesc.llvm_func_name + ".ll",
                    "w",
                ) as f:
                    f.write(kernel_cres.library._final_module.__str__())

        except errors.TypingError as e:
            self._failed_cache[key] = e
            return _KernelCompileResult(False, e, None)
        else:
            return _KernelCompileResult(True, kernel_cres, kernel_module)


class KernelDispatcher(Dispatcher):
    targetdescr = dpex_kernel_target
    _fold_args = False

    Dispatcher._impl_kinds["kernel"] = _KernelCompiler

    def __init__(
        self,
        pyfunc,
        debug_flags=None,
        compile_flags=None,
        specialization_sigs=None,
        enable_cache=True,
        locals={},
        targetoptions={},
        impl_kind="kernel",
        pipeline_class=kernel_compiler.KernelCompiler,
    ):
        targetoptions["nopython"] = True
        targetoptions["experimental"] = True

        self._kernel_name = pyfunc.__name__
        self._range = None
        self._ndrange = None

        self.typingctx = self.targetdescr.typing_context
        self.targetctx = self.targetdescr.target_context

        pysig = utils.pysignature(pyfunc)
        arg_count = len(pysig.parameters)

        self.overloads = OrderedDict()

        can_fallback = not targetoptions.get("nopython", False)

        _DispatcherBase.__init__(
            self,
            arg_count,
            pyfunc,
            pysig,
            can_fallback,
            exact_match_required=False,
        )
        # XXX: What does this function do exactly?
        functools.update_wrapper(self, pyfunc)

        self.targetoptions = targetoptions
        self.locals = locals
        self._cache = NullCache()
        compiler_class = self._impl_kinds[impl_kind]
        self._impl_kind = impl_kind
        self._compiler = compiler_class(
            pyfunc, self.targetdescr, targetoptions, locals, pipeline_class
        )
        self._cache_hits = Counter()
        self._cache_misses = Counter()

        self._type = types.Dispatcher(self)
        self.typingctx.insert_global(self, self._type)

        # Remember target restriction
        self._required_target_backend = targetoptions.get("target_backend")

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

    def add_overload(self, cres, kernel_module):
        args = tuple(cres.signature.args)
        self.overloads[args] = kernel_module

    def compile(self, sig) -> _KernelCompileResult:
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
                # Don't recompile if signature already exists
                existing = self.overloads.get(tuple(args))
                if existing is not None:
                    return existing

                # FIXME: Enable caching
                # Add code to enable on disk caching of a binary spirv kernel
                self._cache_misses[sig] += 1
                ev_details = dict(
                    dispatcher=self,
                    args=args,
                    return_type=return_type,
                )
                with ev.trigger_event("numba_dpex:compile", data=ev_details):
                    try:
                        kcres: _KernelCompileResult = self._compiler.compile(
                            args, return_type
                        )
                    except errors.ForceLiteralArg as e:

                        def folded(args, kws):
                            return self._compiler.fold_argument_types(
                                args, kws
                            )[1]

                        raise e.bind_fold_arguments(folded)
                    self.add_overload(kcres.cres_or_error, kcres.kernel_module)

                # FIXME: enable caching

                return kcres.kernel_module

    def __getitem__(self, args):
        """Square-bracket notation for configuring the global_range and
        local_range settings when launching a kernel on a SYCL queue.

        When a Python function decorated with the @kernel decorator,
        is invoked it creates a KernelLauncher object. Calling the
        KernelLauncher objects ``__getitem__`` function inturn clones the object
        and sets the ``global_range`` and optionally the ``local_range``
        attributes with the arguments passed to ``__getitem__``.

        Args:
            args (tuple): A tuple of tuples that specify the global and
            optionally the local range for the kernel execution. If the
            argument is a two-tuple of tuple, then it is assumed that both
            global and local range options are specified. The first entry is
            considered to be the global range and the second the local range.

            If only a single tuple value is provided, then the kernel is
            launched with only a global range and the local range configuration
            is decided by the SYCL runtime.

        Returns:
            KernelLauncher: A clone of the KernelLauncher object, but with the
            global_range and local_range attributes initialized.
        """

        if isinstance(args, Range):
            self._range = args
        elif isinstance(args, NdRange):
            self._ndrange = args
        else:
            # FIXME: Improve error message
            raise InvalidKernelLaunchArgsError(kernel_name=self._kernel_name)

        return self

    def __call__(self, *args, **kw_args):
        """Functor to launch a kernel."""

        raise NotImplementedError
