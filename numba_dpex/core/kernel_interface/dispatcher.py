# SPDX-FileCopyrightText: 2022 Intel Corporation
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
    ComputeFollowsDataInferenceError,
    ExecutionQueueInferenceError,
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


def get_ordered_arg_access_types(pyfunc, access_types):
    """Deprecated and to be removed in next release."""
    # Construct a list of access type of each arg according to their position
    ordered_arg_access_types = []
    sig = signature(pyfunc, follow_wrapped=False)
    for idx, arg_name in enumerate(sig.parameters):
        if access_types:
            for key in access_types:
                if arg_name in access_types[key]:
                    ordered_arg_access_types.append(key)
        if len(ordered_arg_access_types) <= idx:
            ordered_arg_access_types.append(None)

    return ordered_arg_access_types


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
        array_access_specifiers=None,
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

        if array_access_specifiers:
            warn(
                "Access specifiers apply only to NumPy ndarrays. "
                + "Support for NumPy ndarray objects as kernel arguments "
                + "and access specifiers flags is deprecated. "
                + "Use dpctl.tensor.usm_ndarray based arrays instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.array_access_specifiers = array_access_specifiers

        if debug_flags or config.OPT == 0:
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

    def _chk_compute_follows_data_compliance(self, usm_array_arglist):
        """Check if all the usm ndarray's have the same device.

        Extracts the device filter string from the Numba inferred USMNdArray
        type. Check if the devices corresponding to the filter string are
        equivalent and return a ``dpctl.SyclDevice`` object corresponding to the
        common filter string.

        If an exception occurred in creating a ``dpctl.SyclDevice``, or the
        devices are not equivalent then returns None.

        Args:
            usm_array_arglist : A list of usm_ndarray types specified as
            arguments to the kernel.

        Returns:
            A ``dpctl.SyclDevice`` object if all USMNdArray have same device, or
            else None is returned.
        """

        queue = None

        for usm_array in usm_array_arglist:
            _queue = usm_array.queue
            if not queue:
                queue = _queue
            else:
                if _queue != queue:
                    return None

        return queue

    def _determine_kernel_launch_queue(self, args, argtypes):
        """Determines the queue where the kernel is to be launched.

        The execution queue is derived using the following algorithm. In future,
        support for ``numpy.ndarray`` and ``dpctl.device_context`` is to be
        removed and queue derivation will follows Python Array API's
        "compute follows data" logic.

        Check if there are array arguments.
        True:
          Check if all array arguments are of type numpy.ndarray
          (numba.types.Array)
              True:
                  Check if the kernel was invoked from within a
                  dpctl.device_context.
                  True:
                      Provide a deprecation warning for device_context use and
                      point to using dpctl.tensor.usm_ndarray or dpnp.ndarray

                      return dpctl.get_current_queue
                  False:
                      Raise ExecutionQueueInferenceError
              False:
                  Check if all of the arrays are USMNdarray
                      True:
                          Check if execution queue could be inferred using
                          compute follows data rules
                          True:
                              return the compute follows data inferred queue
                          False:
                              Raise ComputeFollowsDataInferenceError
                      False:
                          Raise ComputeFollowsDataInferenceError
        False:
          Check if the kernel was invoked from within a dpctl.device_context.
            True:
                Provide a deprecation warning for device_context use and
                point to using dpctl.tensor.usm_ndarray of dpnp.ndarray

                return dpctl.get_current_queue
            False:
                Raise ExecutionQueueInferenceError

        Args:
            args : A list of arguments passed to the kernel stored in the
            launcher.
            argtypes : The Numba inferred type for each argument.

        Returns:
            A queue the common queue used to allocate the arrays. If no such
            queue exists, then raises an Exception.

        Raises:
            ComputeFollowsDataInferenceError: If the queue could not be inferred
                using compute follows data rules.
            ExecutionQueueInferenceError: If the queue could not be inferred
                using the dpctl queue manager.
        """

        # FIXME: The args parameter is not needed once numpy support is removed

        # Needed as USMNdArray derives from Array
        array_argnums = [
            i
            for i, _ in enumerate(args)
            if isinstance(argtypes[i], NpArrayType)
            and not isinstance(argtypes[i], USMNdArray)
        ]
        usmarray_argnums = [
            i for i, _ in enumerate(args) if isinstance(argtypes[i], USMNdArray)
        ]

        # if usm and non-usm array arguments are getting mixed, then the
        # execution queue cannot be inferred using compute follows data rules.
        if array_argnums and usmarray_argnums:
            raise ComputeFollowsDataInferenceError(
                array_argnums, usmarray_argnum_list=usmarray_argnums
            )
        elif array_argnums and not usmarray_argnums:
            if dpctl.is_in_device_context():
                warn(
                    "Support for dpctl.device_context to specify the "
                    + "execution queue is deprecated. "
                    + "Use dpctl.tensor.usm_ndarray based array "
                    + "containers instead. ",
                    DeprecationWarning,
                    stacklevel=2,
                )
                warn(
                    "Support for NumPy ndarray objects as kernel arguments is "
                    + "deprecated. Use dpctl.tensor.usm_ndarray based array "
                    + "containers instead. ",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return dpctl.get_current_queue()
            else:
                raise ExecutionQueueInferenceError(self.kernel_name)
        elif usmarray_argnums and not array_argnums:
            if dpctl.is_in_device_context():
                warn(
                    "dpctl.device_context ignored as the kernel arguments "
                    + "are dpctl.tensor.usm_ndarray based array containers."
                )
            usm_array_args = [
                argtype
                for i, argtype in enumerate(argtypes)
                if i in usmarray_argnums
            ]

            queue = self._chk_compute_follows_data_compliance(usm_array_args)

            if not queue:
                raise ComputeFollowsDataInferenceError(
                    self.kernel_name, usmarray_argnum_list=usmarray_argnums
                )

            return queue
        else:
            if dpctl.is_in_device_context():
                warn(
                    "Support for dpctl.device_context to specify the "
                    + "execution queue is deprecated. "
                    + "Use dpctl.tensor.usm_ndarray based array "
                    + "containers instead. ",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return dpctl.get_current_queue()
            else:
                raise ExecutionQueueInferenceError(self.kernel_name)

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
                "The current syntax for specification of kernel lauch "
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
        exec_queue = self._determine_kernel_launch_queue(args, argtypes)
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
            access_specifiers_list=self.array_access_specifiers,
        )

        # Make sure the kernel lauch range/nd_range are sane
        self._check_ranges(exec_queue.sycl_device)

        exec_queue.submit(
            sycl_kernel,
            packer.unpacked_args,
            self._global_range,
            self._local_range,
        )

        exec_queue.wait()

        # TODO remove once NumPy support is removed
        packer.repacked_args
