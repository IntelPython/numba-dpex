# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from inspect import signature
from warnings import warn

import dpctl
import dpctl.program as dpctl_prog
from numba.core import utils
from numba.core.types import Array as ArrayType

from numba_dpex import config
from numba_dpex.core.caching import LRUCache, NullCache
from numba_dpex.core.descriptor import dpex_target
from numba_dpex.core.exceptions import (
    ComputeFollowsDataInferenceError,
    ExecutionQueueInferenceError,
    IllegalRangeValueError,
    InvalidKernelLaunchArgsError,
    SUAIProtocolError,
    UnknownGlobalRangeError,
    UnsupportedBackendError,
    UnsupportedNumberOfRangeDimsError,
)
from numba_dpex.core.kernel_interface.arg_pack_unpacker import Packer
from numba_dpex.core.kernel_interface.spirv_kernel import SpirvKernel
from numba_dpex.dpctl_iface import USMNdArrayType


def get_ordered_arg_access_types(pyfunc, access_types):
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


class Dispatcher(object):
    """Creates a Kernel object from a @kernel decorated function and enqueues
    the Kernel object on a specified device.
    """

    # The list of SYCL backends supported by the Dispatcher
    _supported_backends = ["opencl", "level_zero"]

    def __init__(
        self,
        pyfunc,
        debug_flags=None,
        compile_flags=None,
        array_access_specifiers=None,
        enable_cache=True,
    ):
        self.typingctx = dpex_target.typing_context
        self.pyfunc = pyfunc
        self.debug_flags = debug_flags
        self.compile_flags = compile_flags
        self.kernel_name = pyfunc.__name__
        # TODO: To be removed once the__getitem__ is removed
        self._global_range = None
        self._local_range = None
        # caching related attributes
        if enable_cache:
            self._cache = LRUCache()
        else:
            self._cache = NullCache()
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

    def enable_caching(self):
        self._cache = LRUCache()

    @property
    def cache(self):
        return self._cache

    @property
    def cache_hits(self):
        return self._cache_hits

    def _check_range(self, range, device):
        if not isinstance(range, (tuple, list)):
            raise IllegalRangeValueError(self.kernel_name)

        max_work_item_dims = device.max_work_item_dims

        if len(range) > max_work_item_dims:
            raise UnsupportedNumberOfRangeDimsError(
                kernel_name=self.kernel_name,
                ndims=len(range),
                max_work_item_dims=max_work_item_dims,
            )

    def _check_ndrange(self, global_range, local_range, device):
        # for dim, size in enumerate(val):
        #     if val[dim] > work_item_sizes[dim]:
        #         raise UnsupportedWorkItemSizeError(
        #             kernel_name=self.kernel_name,
        #             dim=dim,
        #             requested_work_items=val[dim],
        #             supported_work_items=work_item_sizes[dim],
        #         )
        pass

    def _determine_compute_follows_data_queue(self, usm_array_list):
        """Determine the execution queue for the list of usm array args using
        compute follows data programming model.

        Uses ``dpctl.utils.get_execution_queue()`` to check if the list of
        queues belonging to the usm_ndarrays are equivalent. If the queues are
        equivalent, then returns the queue. If the queues are not equivalent
        then returns None.

        Args:
            usm_array_list : A list of usm_ndarray objects

        Returns:
            A queue the common queue used to allocate the arrays. If no such
            queue exists, then returns None.
        """
        queues = []
        for usm_array in usm_array_list:
            try:
                q = usm_array.__sycl_usm_array_interface__["syclobj"]
                queues.append(q)
            except:
                raise SUAIProtocolError(self.kernel_name, usm_array)
        return dpctl.utils.get_execution_queue(queues)

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
        # Temporary workaround as USMNdArrayType derives from Array
        array_argnums = [
            i
            for i, arg in enumerate(args)
            if isinstance(argtypes[i], ArrayType)
            and not isinstance(argtypes[i], USMNdArrayType)
        ]
        usmarray_argnums = [
            i
            for i, arg in enumerate(args)
            if isinstance(argtypes[i], USMNdArrayType)
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
                arg for i, arg in enumerate(args) if i in usmarray_argnums
            ]
            queue = self._determine_compute_follows_data_queue(usm_array_args)
            if not queue:
                raise ComputeFollowsDataInferenceError(
                    self.kernel_name, usmarray_argnum_list=usmarray_argnums
                )
            else:
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

        .. deprecated:: 0.19
            Use :func:`KernelLauncher.execute` instead.
        """

        warn(
            "The [] (__getitem__) method to set global and local ranges for "
            + "launching a kernel is deprecated. "
            + "Use the execute function instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        nargs = len(args)
        # Check if the kernel launch arguments are sane.
        if nargs < 1:
            raise UnknownGlobalRangeError(kernel_name=self.kernel_name)
        elif nargs > 2:
            raise InvalidKernelLaunchArgsError(
                kernel_name=self.kernel_name, args=args
            )
        self._global_range = args[0]
        if nargs == 2 and args[1] != []:
            self._local_range = args[1]
        else:
            self._local_range = None

        return copy.copy(self)

    def _get_ranges(self, global_range, local_range, device):
        """_summary_

        Args:
            global_range (_type_): _description_
            local_range (_type_): _description_

        Raises:
            UnknownGlobalRangeError: _description_
        """
        if global_range:
            if self._global_range:
                warn(
                    "Ignoring the previously set value of global_range and "
                    + "using the value specified at the kernel call site."
                )
        else:
            if self._global_range:
                warn(
                    "Use of __getitem__ to set the global_range attribute is "
                    + 'deprecated. Use the keyword argument "global_range" of '
                    + "__call__ method to set the attribute."
                )
                global_range = self._global_range
            else:
                raise UnknownGlobalRangeError(self.kernel_name)

        if local_range:
            if self._local_range:
                warn(
                    "Ignoring the previously set value of local_range and "
                    + "using the value specified at the kernel call site.."
                )
        else:
            if self._local_range:
                warn(
                    "Use of __getitem__ to set the local_range attribute is "
                    + 'deprecated. Use the keyword argument "local_range" of '
                    + "__call__ method to set the attribute."
                )
                local_range = self._local_range
            else:
                local_range = None
                warn(
                    "Kernel to be submitted without a local range letting "
                    + "the SYCL runtime select a local range. The behavior "
                    + "can lead to suboptimal performance in certain cases. "
                    + "Consider setting the local range value for the kernel "
                    + "execution.\n"
                    + "The local_range keyword may be made a required argument "
                    + "in the future."
                )

        if isinstance(global_range, int):
            global_range = [global_range]

        # If only global range value is provided, then the kernel is invoked
        # over an N-dimensional index space defined by a SYCL range<N>, where
        # N is one, two or three.
        # If both local and global range values are specified the kernel is
        # invoked using a SYCL nd_range
        if global_range and not local_range:
            self._check_range(global_range, device)
            # FIXME:[::-1] is done as OpenCL and SYCl have different orders when it
            # comes to specifying dimensions.
            global_range = list(global_range)[::-1]
        else:
            if isinstance(local_range, int):
                local_range = [local_range]
            self._check_ndrange(
                global_range=global_range,
                local_range=local_range,
                device=device,
            )
            global_range = list(global_range)[::-1]
            local_range = list(local_range)[::-1]

        return (global_range, local_range)

    def __call__(self, *args, global_range=None, local_range=None):
        """_summary_

        Args:
            global_range (_type_): _description_
            local_range (_type_): _description_.
        """
        argtypes = [self.typingctx.resolve_argument_type(arg) for arg in args]

        exec_queue = self._determine_kernel_launch_queue(args, argtypes)
        backend = exec_queue.backend
        device_type = exec_queue.sycl_device.device_type

        if exec_queue.backend not in [
            dpctl.backend_type.opencl,
            dpctl.backend_type.level_zero,
        ]:
            raise UnsupportedBackendError(
                self.kernel_name, backend, Dispatcher._supported_backends
            )

        # TODO: Refactor after __getitem__ is removed
        global_range, local_range = self._get_ranges(
            global_range, local_range, exec_queue.sycl_device
        )

        # TODO: Enable caching of kernels, but do it using Numba's caching
        # machinery

        # load the kernel from cache
        sig = utils.pysignature(self.pyfunc)
        key = LRUCache.build_key(
            sig,
            self.pyfunc,
            dpex_target.target_context.codegen(),
            backend=backend,
            device_type=device_type,
        )
        artifact = self._cache.get(key)
        if artifact is not None:
            device_driver_ir_module, kernel_module_name = artifact
            self._cache_hits += 1
        else:
            kernel = SpirvKernel(self.pyfunc, self.kernel_name)

            kernel.compile(
                arg_types=argtypes,
                debug=self.debug_flags,
                extra_compile_flags=self.compile_flags,
            )
            device_driver_ir_module = kernel.device_driver_ir_module
            kernel_module_name = kernel.module_name

            key = LRUCache.build_key(
                sig,
                self.pyfunc,
                kernel.target_context.codegen(),
                backend=backend,
                device_type=device_type,
            )
            self._cache.put(key, (device_driver_ir_module, kernel_module_name))

        # create a sycl::KernelBundle
        kernel_bundle = dpctl_prog.create_program_from_spirv(
            exec_queue,
            device_driver_ir_module,
            " ".join(self._create_sycl_kernel_bundle_flags),
        )
        #  get the sycl::kernel
        sycl_kernel = kernel_bundle.get_sycl_kernel(kernel_module_name)

        packer = Packer(
            kernel_name=self.kernel_name,
            arg_list=args,
            argty_list=argtypes,
            queue=exec_queue,
            access_specifiers_list=self.array_access_specifiers,
        )

        exec_queue.submit(
            sycl_kernel,
            packer.unpacked_args,
            global_range,
            local_range,
        )

        exec_queue.wait()

        # TODO remove once NumPy support is removed
        packer.repacked_args
