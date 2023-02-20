# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from warnings import warn

import dpctl
from numba.core.types import Array as NpArrayType

from numba_dpex.core.exceptions import (
    ComputeFollowsDataInferenceError,
    ExecutionQueueInferenceError,
)
from numba_dpex.core.types import USMNdArray


def chk_compute_follows_data_compliance(usm_array_arglist):
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


def determine_kernel_launch_queue(args, argtypes, kernel_name):
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
            raise ExecutionQueueInferenceError(kernel_name)
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

        queue = chk_compute_follows_data_compliance(usm_array_args)

        if not queue:
            raise ComputeFollowsDataInferenceError(
                kernel_name, usmarray_argnum_list=usmarray_argnums
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
            raise ExecutionQueueInferenceError(kernel_name)
