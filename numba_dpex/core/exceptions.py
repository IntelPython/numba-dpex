# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""The module defines the custom error classes used in numba_dpex.
"""

from warnings import warn


class KernelHasReturnValueError(Exception):
    """Exception raised when a kernel function is defined with a return
    statement.


    @numba_dpex.kernel does not allow users to return any value. The
    restriction is inline with the general ABI for device functions in OpenCL,
    CUDA, and SYCL.

    Args:
        kernel_name: Name of the kernel function that caused the error.
        return_type: Numba type representing the return value specified for
        the kernel function.
    """

    def __init__(self, kernel_name, return_type, sig=None) -> None:
        self.return_type = return_type
        if sig:
            self.message = (
                f'Specialized kernel signature "{sig}" has a return value '
                f'of type "{return_type}". '
                "A numba-dpex kernel must have a void return type."
            )
        else:
            self.message = (
                f'Kernel "{kernel_name}" has a return value '
                f'of type "{return_type}". '
                "A numba-dpex kernel must have a void return type."
            )

        super().__init__(self.message)


class InvalidKernelLaunchArgsError(Exception):
    """Exception raised when a kernel is dispatched with insufficient or
    incorrect launch arguments.

    A global range value is needed to submit a kernel to an execution queue. The
    global range provides the number of work items that the SYCL runtime should
    create to execute the kernel. The exception is raised if a kernel is
    dispatched without specifying a valid global range.

    Args:
        kernel_name (str): The kernel function name.
    """

    def __init__(self, kernel_name):
        warn(
            "The InvalidKernelLaunchArgsError class is deprecated, and will "
            + "be removed once kernel launching using __getitem__ for the "
            + "KernelLauncher class is removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.message = (
            "Invalid launch arguments specified for launching the Kernel "
            f'"{kernel_name}". Launch arguments can only be a tuple '
            "specifying the global range or a tuple of tuples specifying "
            "global and local ranges."
        )
        super().__init__(self.message)


class UnknownGlobalRangeError(Exception):
    """Exception raised when a kernel is launched without specifying a global
    range.

    Args:
        kernel_name (str): The kernel function name.
    """

    def __init__(self, kernel_name) -> None:
        self.message = (
            "No global range specified for launching the Kernel "
            f'"{kernel_name}". '
            "A valid global range tuple is needed to launch a kernel."
        )
        super().__init__(self.message)


class IllegalRangeValueError(Exception):
    def __init__(self, kernel_name) -> None:
        self.message = (
            f"Kernel {kernel_name} cannot be dispatched with the "
            "specified range. The range should be specified as a list, tuple, "
            "or an int."
        )
        super().__init__(self.message)


class UnsupportedNumberOfRangeDimsError(Exception):
    def __init__(self, kernel_name, ndims, max_work_item_dims) -> None:
        self.message = (
            f"Specified range for kernel {kernel_name} has {ndims} dimensions, "
            f"the device supports only {max_work_item_dims} dimensional "
            "ranges."
        )
        super().__init__(self.message)


class UnsupportedWorkItemSizeError(Exception):
    """

    Args:
        Exception (_type_): _description_
    """

    def __init__(
        self, kernel_name, dim, requested_work_items, supported_work_items
    ) -> None:
        self.message = (
            f"Attempting to launch kernel {kernel_name} with "
            f"{requested_work_items} work items in dimension {dim} is not "
            f"supported. The device supports only {supported_work_items} "
            f"work items for dimension {dim}."
        )
        super().__init__(self.message)


class ComputeFollowsDataInferenceError(Exception):
    """Exception raised when an execution queue for a given array expression or
    a kernel function could not be deduced using the compute-follows-data
    programming model.

    Compute-follows-data is a programming model that determines the execution
    device or queue of an array expression or kernel based on the arrays that
    are on the right hand side of the expression or are the arguments to the
    kernel function. The execution queue is deduced based on the device on
    which the array operands were allocated. Computation is required to occur
    on the same device where the arrays currently reside.

    A ComputeFollowsDataInferenceError is raised when the execution queue using
    compute-follows-data rules could not be deduced. It may happen when arrays
    that have a device attribute such as ``dpctl.tensor.usm_ndarray`` are mixed
    with host arrays such as ``numpy.ndarray``. The error may also be raised if
    the array operands are allocated on different devices.

    Args:
        kernel_name : Name of the kernel function for which the error occurred.
        ndarray_argnum_list: The list of ``numpy.ndarray`` arguments identified
        by the argument position that caused the error.
        usmarray_argnum_list: The list of ``dpctl.tensor.usm_ndarray`` arguments
        identified by the argument position that caused the error.
    """

    def __init__(
        self, kernel_name, ndarray_argnum_list=None, *, usmarray_argnum_list
    ) -> None:
        if ndarray_argnum_list and usmarray_argnum_list:
            ndarray_args = ",".join([str(i) for i in ndarray_argnum_list])
            usmarray_args = ",".join([str(i) for i in usmarray_argnum_list])
            self.message = (
                f'Kernel "{kernel_name}" has arguments of both usm_ndarray and '
                "non-usm_ndarray types. Mixing of arguments of different "
                "array types is disallowed. "
                f"Arguments {ndarray_args} are non-usm arrays, "
                f"and arguments {usmarray_args} are usm arrays."
            )
        elif usmarray_argnum_list:
            usmarray_args = ",".join([str(i) for i in usmarray_argnum_list])
            self.message = (
                f'Execution queue for kernel "{kernel_name}" could '
                "be deduced using compute follows data programming model. The "
                f"usm_ndarray arguments {usmarray_args} were not allocated "
                "on the same queue."
            )
        super().__init__(self.message)


class ExecutionQueueInferenceError(Exception):
    """Exception raised when an execution queue could not be deduced for NumPy
    ndarray kernel arguments.

    Args:
        kernel_name (str): Name of kernel where the error was raised.

    .. deprecated:: 0.19
    """

    def __init__(self, kernel_name) -> None:
        warn(
            "The ExecutionQueueInferenceError class is deprecated, and will "
            + "be removed once support for NumPy ndarrays as kernel arguments "
            + "is removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.message = (
            f'Kernel "{kernel_name}" was called with NumPy ndarray arguments '
            "outside a dpctl.device_context. The execution queue to be used "
            "could not be deduced."
        )
        super().__init__(self.message)


class UnsupportedBackendError(Exception):
    """Exception raised when the target device is not supported by dpex.

    Presently only L0 and OpenCL devices are supported by numba_dpex. If the
    provided execution queue is for any other SYCL platform (e.g., Cuda) then
    it results in an UnsupportedBackendError.

    Args:
        kernel_name (str): Name of kernel where the error was raised.
        backend (str): name of the unsupported backend.
        supported_backends (List): The list of supported backends.
    """

    def __init__(self, kernel_name, backend, supported_backends) -> None:
        supported_backends_joined = ",".join(supported_backends)
        self.message = (
            f'Kernel "{kernel_name}" cannot be compiled for "{backend}". '
            f'Only "{supported_backends_joined}" backends are supported.'
        )
        super().__init__(self.message)


class UncompiledKernelError(Exception):
    """Exception raised when an attribute of a KernelInterface object is
    accessed before the object is compiled.

    Args:
        kernel_name (str): Name of kernel where the error was raised.
    """

    def __init__(self, kernel_name) -> None:
        self.message = (
            f'No LLVM module for kernel "{kernel_name}". '
            "Kernel might not be compiled yet."
        )
        super().__init__(self.message)


class UnreachableError(Exception):
    """Internal compiler error when an unreachable branch is taken somewhere in
    the compiler code.
    """

    def __init__(self) -> None:
        import sys
        import traceback

        _, _, tb = sys.exc_info()
        if tb:
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            self.message = (
                f"Attempt to execute a unreachable code section on line {line} "
                f"in statement {text} in function {func} in the module "
                f"{filename}."
            )
        else:
            self.message = "Unreachable code executed."
        super().__init__(self.message)


class UnsupportedKernelArgumentError(Exception):
    """Exception raised when the type of a kernel argument is not supported by
    the compiler.

    Args:
        kernel_name (str): Name of kernel where the error was raised.
    """

    def __init__(self, type, value, kernel_name) -> None:
        self.message = (
            f'Argument {value} passed to kernel "{kernel_name}" is of an '
            f"unsupported type ({type})."
        )
        super().__init__(self.message)


class SUAIProtocolError(Exception):
    """Exception raised when an array-like object passed to a kernel is
    neither a NumPy array nor does it implement the __sycl_usm_array_interface__
    attribute.

    Args:
        kernel_name (str): Name of kernel where the error was raised.
        arg: Array-like object
    """

    def __init__(self, kernel_name, arg) -> None:
        self.message = (
            f'Array-like argument {arg} passed to kernel "{kernel_name}" '
            "is neither a NumPy array nor implement the "
            "__sycl_usm_array_interface__."
        )
        super().__init__(self.message)


class UnsupportedAccessQualifierError(Exception):
    """Exception raised when an illegal access specifier value is specified for
    a NumPy array argument passed to a kernel.

    Args:
        kernel_name (str): Name of kernel where the error was raised.
        array_val: name of the array argument with the illegal access specifier.
        illegal_access_type (str): The illegal access specifier string.
        legal_access_list (str): Joined string for the legal access specifiers.
    """

    def __init__(
        self, kernel_name, array_val, illegal_access_type, legal_access_list
    ) -> None:
        self.message = f"Invalid access type {illegal_access_type} applied to "
        f'array {array_val} argument passed to kernel "{kernel_name}". '
        f"Legal access specifiers are {legal_access_list}."

        super().__init__(self.message)


class UnsupportedCompilationModeError(Exception):
    def __init__(self) -> None:
        self.message = (
            'The dpex compiler does not support the "force_pyobject" setting.'
        )
        super().__init__(self.message)


class InvalidKernelSpecializationError(Exception):
    def __init__(
        self, kernel_name, invalid_sig, unsupported_argnum_list
    ) -> None:
        unsupported = ",".join([str(i) for i in unsupported_argnum_list])
        self.message = f"Kernel {kernel_name} cannot be specialized for "
        f'"{invalid_sig}". Arguments {unsupported} are not supported.'

        super().__init__(self.message)
