# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba_dpex.core.exceptions import ExecutionQueueInferenceError
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

    The execution queue is derived following Python Array API's
    "compute follows data" programming model.

    Args:
        argtypes : The Numba inferred type for each argument.
        kernel_name : The name of the kernel function

    Returns:
        A queue the common queue used to allocate the arrays. If no such
        queue exists, then raises an Exception.

    Raises:
        ExecutionQueueInferenceError: If the queue could not be inferred.
    """

    usmarray_argnums = [
        i for i, _ in enumerate(args) if isinstance(argtypes[i], USMNdArray)
    ]

    usm_array_args = [
        argtype for i, argtype in enumerate(argtypes) if i in usmarray_argnums
    ]

    queue = chk_compute_follows_data_compliance(usm_array_args)

    if not queue:
        raise ExecutionQueueInferenceError(
            kernel_name, usmarray_argnum_list=usmarray_argnums
        )

    return queue
