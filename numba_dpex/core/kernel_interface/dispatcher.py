# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from inspect import signature
from warnings import warn

import dpctl
import dpctl.program as dpctl_prog
from numba.core import sigutils
from numba.core.types import Array as NpArrayType
from numba.core.types import void

from numba_dpex import NdRange, Range, config
from numba_dpex.core.caching import LRUCache, NullCache
from numba_dpex.core.descriptor import dpex_kernel_target
from numba_dpex.core.exceptions import (
    IllegalRangeValueError,
    InvalidKernelLaunchArgsError,
    InvalidKernelSpecializationError,
    KernelHasReturnValueError,
    MissingSpecializationError,
    UnknownGlobalRangeError,
    UnmatchedNumberOfRangeDimsError,
    UnsupportedBackendError,
    UnsupportedGroupWorkItemSizeError,
    UnsupportedNumberOfRangeDimsError,
    UnsupportedWorkItemSizeError,
)
from numba_dpex.core.kernel_interface.arg_pack_unpacker import Packer
from numba_dpex.core.kernel_interface.spirv_kernel import SpirvKernel
from numba_dpex.core.types import USMNdArray
from numba_dpex.core.utils import (
    build_key,
    create_func_hash,
    strip_usm_metadata,
)

from .utils import determine_kernel_launch_queue


class JitKernel:
    """Functor to wrap a kernel function and JIT compile and dispatch it to a
    specified SYCL queue.

    A JitKernel is returned by the kernel decorator and wraps an instance of a
    device kernel function. A device kernel function is specialized for a
    backend may represent a binary object in a lower-level IR. Currently, only
    SPIR-V binary format device functions for level-zero and opencl backends
    are supported.

    """

    # The list of SYCL backends supported by the Dispatcher
    _supported_backends = ["opencl", "level_zero"]

    def __init__(
        self,
        pyfunc,
        debug_flags=None,
        compile_flags=None,
        specialization_sigs=None,
        enable_cache=True,
    ):
        self.typingctx = dpex_kernel_target.typing_context
        self.pyfunc = pyfunc
        self.debug_flags = debug_flags
        self.compile_flags = compile_flags
        self.kernel_name = pyfunc.__name__

        self._global_range = None
        self._local_range = None

        self._func_hash = create_func_hash(pyfunc)

        # caching related attributes
        if not config.ENABLE_CACHE:
            self._cache = NullCache()
            self._kernel_bundle_cache = NullCache()
        elif enable_cache:
            self._cache = LRUCache(
                name="SPIRVKernelCache",
                capacity=config.CACHE_SIZE,
                pyfunc=self.pyfunc,
            )
            self._kernel_bundle_cache = LRUCache(
                name="KernelBundleCache",
                capacity=config.CACHE_SIZE,
                pyfunc=self.pyfunc,
            )
        else:
            self._cache = NullCache()
            self._kernel_bundle_cache = NullCache()
        self._cache_hits = 0

        if debug_flags or config.DPEX_OPT == 0:
            # if debug is ON we need to pass additional
            # flags to igc.
            self._create_sycl_kernel_bundle_flags = ["-g", "-cl-opt-disable"]
        else:
            self._create_sycl_kernel_bundle_flags = []

        # Specialization of kernel based on signatures. If specialization
        # signatures are found, they are compiled ahead of time and cached.
        if specialization_sigs:
            self._has_specializations = True
            self._specialization_cache = LRUCache(
                name="SPIRVKernelSpecializationCache",
                capacity=config.CACHE_SIZE,
                pyfunc=self.pyfunc,
            )
            for sig in specialization_sigs:
                self._specialize(sig)
            if self._specialization_cache.size() == 0:
                raise AssertionError(
                    "JitKernel could not be specialized for signatures: "
                    + specialization_sigs
                )
        else:
            self._has_specializations = False
            self._specialization_cache = NullCache()

    @property
    def cache(self):
        return self._cache

    @property
    def cache_hits(self):
        return self._cache_hits

    def _compile_and_cache(self, argtypes, cache, key=None):
        """Helper function to compile the Python function or Numba FunctionIR
        object passed to a JitKernel and store it in an internal cache.
        """
        # We always compile the kernel using the dpex_target.
        typingctx = dpex_kernel_target.typing_context
        targetctx = dpex_kernel_target.target_context

        kernel = SpirvKernel(self.pyfunc, self.kernel_name)
        kernel.compile(
            args=argtypes,
            typing_ctx=typingctx,
            target_ctx=targetctx,
            debug=self.debug_flags,
            compile_flags=self.compile_flags,
        )

        device_driver_ir_module = kernel.device_driver_ir_module
        kernel_module_name = kernel.module_name

        if not key:
            stripped_argtypes = strip_usm_metadata(argtypes)
            codegen_magic_tuple = kernel.target_context.codegen().magic_tuple()
            key = build_key(
                stripped_argtypes, codegen_magic_tuple, self._func_hash
            )

        cache.put(key, (device_driver_ir_module, kernel_module_name))

        return device_driver_ir_module, kernel_module_name

    def _specialize(self, sig):
        """Compiles a device kernel ahead of time based on provided signature.

        Args:
            sig: The signature on which the kernel is to be specialized.
        """

        argtypes, return_type = sigutils.normalize_signature(sig)

        # Check if signature has a non-void return type
        if return_type and return_type != void:
            raise KernelHasReturnValueError(
                kernel_name=None, return_type=return_type, sig=sig
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
                kernel_name=self.kernel_name,
                invalid_sig=sig,
                unsupported_argnum_list=unsupported_argnum_list,
            )

        self._compile_and_cache(
            argtypes=argtypes,
            cache=self._specialization_cache,
        )

    def _check_size(self, dim, size, size_limit):
        """Checks if the range value is sane based on the number of work items
        supported by the device.
        """

        if size > size_limit:
            raise UnsupportedWorkItemSizeError(
                kernel_name=self.kernel_name,
                dim=dim,
                requested_work_items=size,
                supported_work_items=size_limit,
            )

    def _check_range(self, range, device):
        """Checks if the requested range to launch the kernel is valid.

        Range is checked against the number of dimensions and if the range
        argument is specified as a valid list of tuple.
        """

        if not (
            range
            and isinstance(range, list)
            and all(isinstance(v, int) for v in range)
        ):
            raise IllegalRangeValueError(self.kernel_name)

        max_work_item_dims = device.max_work_item_dims

        if len(range) > max_work_item_dims:
            raise UnsupportedNumberOfRangeDimsError(
                kernel_name=self.kernel_name,
                ndims=len(range),
                max_work_item_dims=max_work_item_dims,
            )

    def _check_ndrange(self, global_range, local_range, device):
        """Checks if the specified nd_range (global_range, local_range) is
        legal for a device on which the kernel will be launched.
        """
        self._check_range(local_range, device)

        self._check_range(global_range, device)
        if len(local_range) != len(global_range):
            raise UnmatchedNumberOfRangeDimsError(
                kernel_name=self.kernel_name,
                global_ndims=len(global_range),
                local_ndims=len(local_range),
            )

        for i in range(len(global_range)):
            self._check_size(i, local_range[i], device.max_work_item_sizes[i])
            if global_range[i] % local_range[i] != 0:
                raise UnsupportedGroupWorkItemSizeError(
                    kernel_name=self.kernel_name,
                    dim=i,
                    work_groups=global_range[i],
                    work_items=local_range[i],
                )

    def __getitem__(self, args):
        """Mimic's ``numba.cuda`` square-bracket notation for configuring the
        global_range and local_range settings when launching a kernel on a
        SYCL queue.

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
            # we need inversions, see github issue #889
            self._global_range = list(args)[::-1]
        elif isinstance(args, NdRange):
            # we need inversions, see github issue #889
            self._global_range = list(args.global_range)[::-1]
            self._local_range = list(args.local_range)[::-1]
        else:
            if (
                isinstance(args, tuple)
                and len(args) == 2
                and isinstance(args[0], int)
                and isinstance(args[1], int)
            ):
                warn(
                    "Ambiguous kernel launch paramters. If your data have "
                    + "dimensions > 1, include a default/empty local_range:\n"
                    + "    <function>[(X,Y), numba_dpex.DEFAULT_LOCAL_RANGE]"
                    "(<params>)\n"
                    + "otherwise your code might produce erroneous results.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self._global_range = [args[0]]
                self._local_range = [args[1]]
                return self

            warn(
                "The current syntax for specification of kernel launch "
                + "parameters is deprecated. Users should set the kernel "
                + "parameters through Range/NdRange classes.\n"
                + "Example:\n"
                + "    from numba_dpex import Range,NdRange\n\n"
                + "    # for global range only\n"
                + "    <function>[Range(X,Y)](<parameters>)\n"
                + "    # or,\n"
                + "    # for both global and local ranges\n"
                + "    <function>[NdRange((X,Y), (P,Q))](<parameters>)",
                DeprecationWarning,
                stacklevel=2,
            )

            args = [args] if not isinstance(args, Iterable) else args
            nargs = len(args)

            # Check if the kernel enquing arguments are sane
            if nargs < 1 or nargs > 2:
                raise InvalidKernelLaunchArgsError(kernel_name=self.kernel_name)

            g_range = (
                [args[0]] if not isinstance(args[0], Iterable) else args[0]
            )
            # If the optional local size argument is provided
            l_range = None
            if nargs == 2:
                if args[1] != []:
                    l_range = (
                        [args[1]]
                        if not isinstance(args[1], Iterable)
                        else args[1]
                    )
                else:
                    warn(
                        "Empty local_range calls are deprecated. Please use "
                        "Range/NdRange to specify the kernel launch parameters:"
                        "\n"
                        + "Example:\n"
                        + "    from numba_dpex import Range,NdRange\n\n"
                        + "    # for global range only\n"
                        + "    <function>[Range(X,Y)](<parameters>)\n"
                        + "    # or,\n"
                        + "    # for both global and local ranges\n"
                        + "    <function>[NdRange((X,Y), (P,Q))](<parameters>)",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            if len(g_range) < 1:
                raise IllegalRangeValueError(kernel_name=self.kernel_name)

            # we need inversions, see github issue #889
            self._global_range = list(g_range)[::-1]
            self._local_range = list(l_range)[::-1] if l_range else None

        return self

    def _check_ranges(self, device):
        """Helper to get the validate the global and local range values prior
        to launching a kernel.

        Args:
            device (dpctl.SyclDevice): The device on which to launch the kernel.
        """
        # If only global range value is provided, then the kernel is invoked
        # over an N-dimensional index space defined by a SYCL range<N>, where
        # N is one, two or three.
        # If both local and global range values are specified the kernel is
        # invoked as a SYCL nd_range kernel.

        if not self._global_range:
            raise UnknownGlobalRangeError(self.kernel_name)
        elif self._global_range and not self._local_range:
            self._check_range(self._global_range, device)
        else:
            self._check_ndrange(
                global_range=self._global_range,
                local_range=self._local_range,
                device=device,
            )

    def __call__(self, *args):
        """Functor to launch a kernel."""

        argtypes = [self.typingctx.resolve_argument_type(arg) for arg in args]
        # FIXME: For specialized and ahead of time compiled and cached kernels,
        # the CFD check was already done statically. The run-time check is
        # redundant. We should avoid these checks for the specialized case.
        ty_queue = determine_kernel_launch_queue(
            args, argtypes, self.kernel_name
        )

        # FIXME: We need a better way than having to create a queue every time.
        device = ty_queue.sycl_device
        exec_queue = dpctl.get_device_cached_queue(device)

        backend = exec_queue.backend

        if exec_queue.backend not in [
            dpctl.backend_type.opencl,
            dpctl.backend_type.level_zero,
        ]:
            raise UnsupportedBackendError(
                self.kernel_name, backend, JitKernel._supported_backends
            )

        # Generate key used for cache lookup
        stripped_argtypes = strip_usm_metadata(argtypes)
        codegen_magic_tuple = (
            dpex_kernel_target.target_context.codegen().magic_tuple()
        )
        key = build_key(stripped_argtypes, codegen_magic_tuple, self._func_hash)

        # If the JitKernel was specialized then raise exception if argtypes
        # do not match one of the specialized versions.
        if self._has_specializations:
            artifact = self._specialization_cache.get(key)
            if artifact is not None:
                device_driver_ir_module, kernel_module_name = artifact
            else:
                raise MissingSpecializationError(self.kernel_name, argtypes)
        else:
            artifact = self._cache.get(key)
            # if the kernel was not previously cached, compile it.
            if artifact is not None:
                device_driver_ir_module, kernel_module_name = artifact
                self._cache_hits += 1
            else:
                (
                    device_driver_ir_module,
                    kernel_module_name,
                ) = self._compile_and_cache(
                    argtypes=argtypes, cache=self._cache, key=key
                )

        kernel_bundle_key = build_key(
            stripped_argtypes, codegen_magic_tuple, exec_queue, self._func_hash
        )

        artifact = self._kernel_bundle_cache.get(kernel_bundle_key)

        if artifact is None:
            # create a sycl::KernelBundle
            kernel_bundle = dpctl_prog.create_program_from_spirv(
                exec_queue,
                device_driver_ir_module,
                " ".join(self._create_sycl_kernel_bundle_flags),
            )
            self._kernel_bundle_cache.put(kernel_bundle_key, kernel_bundle)
        else:
            kernel_bundle = artifact

        #  get the sycl::kernel
        sycl_kernel = kernel_bundle.get_sycl_kernel(kernel_module_name)

        packer = Packer(
            kernel_name=self.kernel_name,
            arg_list=args,
            argty_list=argtypes,
            queue=exec_queue,
        )

        # Make sure the kernel launch range/nd_range are sane
        self._check_ranges(exec_queue.sycl_device)

        # TODO: return event that calls wait if no reference to the object if
        # it is possible
        # event = exec_queue.submit(
        exec_queue.submit(
            sycl_kernel,
            packer.unpacked_args,
            self._global_range,
            self._local_range,
        )

        exec_queue.wait()
