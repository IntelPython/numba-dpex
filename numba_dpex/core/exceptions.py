# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
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
                "A kernel must have a void return type."
            )
        else:
            self.message = (
                f'Kernel "{kernel_name}" has a return value '
                f'of type "{return_type}". '
                "A kernel must have a void return type."
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
        self.message = (
            "Invalid global and local range arguments specified for launching "
            f' the Kernel "{kernel_name}". Refer documentation for details.'
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
    """Exception raised when a range used to launch a kernel is not well-formed.

    A non-well-formed range value is either empty or is not a list of ints.

    Args:
        kernel_name (str): The kernel function name.
    """

    def __init__(self, kernel_name) -> None:
        self.message = (
            f"Specified range to launch Kernel {kernel_name} is not "
            "well-formed. A range should be specified as a list, tuple, "
            "or an int using the [] (__getitem__) method of the kernel."
        )
        super().__init__(self.message)


class UnsupportedNumberOfRangeDimsError(Exception):
    """Exception raised when a range is created for a device that has more
    number of dimensions than is supported by the device.

    Args:
        kernel_name (str): The kernel function name.
        ndims (int): Rank of the errant range.
        max_work_item_dims (int): Maximum number of range dimensions supported
        by the device.
    """

    def __init__(self, kernel_name, ndims, max_work_item_dims) -> None:
        self.message = (
            f"Specified range for kernel {kernel_name} has {ndims} dimensions, "
            f"the device supports only {max_work_item_dims} dimensional "
            "ranges."
        )
        super().__init__(self.message)


class UnmatchedNumberOfRangeDimsError(Exception):
    """Exception raised when the global range and local range have different
    number of dimensions or rank.

    Args:
        kernel_name (str): The kernel function name.
        global_ndims (int): Rank of the global range.
        local_ndims (int): Rank of the local range.
    """

    def __init__(self, kernel_name, global_ndims, local_ndims) -> None:
        self.message = (
            f"Specified global_range for kernel {kernel_name} has "
            f"{global_ndims} dimensions, "
            f"while specified local_range with dimensions of {local_ndims} "
            "doesn't match with global_range."
        )
        super().__init__(self.message)


class UnsupportedWorkItemSizeError(Exception):
    """Exception raised when the number of work items requested for a
    specific dimension exceeds the number supported by the device.

    Args:
        kernel_name (str): The kernel function name.
        requested_work_items (int): Number of requested work items.
        supported_work_items (int): Supported number of work items.
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


class UnsupportedGroupWorkItemSizeError(Exception):
    """Exception raised when the value in a specific dimension of a global
    range is not evenly divisible by the value in the corresponding dimension
    in a local range.

    Args:
        kernel_name (str): The kernel function name.
        dim (int): Dimension where the mismatch was identified.
        work_groups (int): Number of requested work groups.
        work_items (int): Number of requested work items in the errant
        dimension of the local range.
    """

    def __init__(self, kernel_name, dim, work_groups, work_items) -> None:
        self.message = (
            f"Attempting to launch kernel {kernel_name} with "
            f"{work_groups} global work groups and {work_items} local work "
            f"items in dimension {dim} is not supported. The global work "
            "groups must be evenly divisible by the local work items."
        )
        super().__init__(self.message)


class ExecutionQueueInferenceError(Exception):
    """Exception raised when an execution queue for a given array expression or
    a kernel function could not be deduced using the compute-follows-data
    programming model.

    Compute-follows-data is a programming model that determines the execution
    device or queue of an array expression or kernel based on the arrays that
    are on the right hand side of the expression or are the arguments to the
    kernel function. The execution queue is deduced based on the device on
    which the array operands were allocated. Computation is required to occur
    on the same device where the arrays currently reside.

    A ExecutionQueueInferenceError is raised when the execution queue using
    compute-follows-data rules could not be deduced. It may happen when arrays
    that have a device attribute such as ``dpctl.tensor.usm_ndarray`` are mixed
    with host arrays such as ``numpy.ndarray``. The error may also be raised if
    the array operands are allocated on different devices.

    Args:
        kernel_name : Name of the kernel function for which the error occurred.
        usmarray_argnum_list: The list of ``dpctl.tensor.usm_ndarray`` arguments
        identified by the argument position that caused the error.
    """

    def __init__(self, kernel_name, *, usmarray_argnum_list) -> None:
        if usmarray_argnum_list is not None:
            usmarray_args = ",".join([str(i) for i in usmarray_argnum_list])
            self.message = (
                f'Execution queue for kernel "{kernel_name}" could '
                "be deduced using compute follows data programming model. The "
                f"usm_ndarray arguments {usmarray_args} were not allocated "
                "on the same queue."
            )
        else:
            self.message = (
                f'Execution queue for kernel "{kernel_name}" could '
                "be deduced using compute follows data programming model. The "
                "kernel has no USMNdArray argument."
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
        type (str): The type of the unsupported argument.
        value (object): The Python object passed as a kernel argument.
        kernel_name (str): Name of kernel where the error was raised.
    """

    def __init__(self, type, value, kernel_name="") -> None:
        self.message = (
            f"Argument {value} passed to kernel {kernel_name} is of an "
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


class UnsupportedCompilationModeError(Exception):
    def __init__(self) -> None:
        self.message = (
            'The dpex compiler does not support the "force_pyobject" setting.'
        )
        super().__init__(self.message)


class InvalidKernelSpecializationError(Exception):
    """Exception raised when a the specialization argument types are not
    supported by the dpex kernel decorator.

    The exception is raised whenever an unsupported kernel argument is
    provided in the specialization signature passed to a dpex kernel decorator
    instance. For example, dpex kernels require arrays to be of USMNdArray type
    and no other Array type, such as NumPy ndarray, are supported. If the
    signature has an non USMNdArray Array type the exception is raised.

    Args:
        kernel_name (str): Name of kernel where the error was raised.
        invalid_sig: Unsupported signature.
        unsupported_argnum_list : The list of argument numbers that are
        unsupported.
    """

    def __init__(
        self, kernel_name, invalid_sig, unsupported_argnum_list
    ) -> None:
        unsupported = ",".join([str(i) for i in unsupported_argnum_list])
        self.message = (
            f"Kernel {kernel_name} cannot be specialized for "
            f'"{invalid_sig}". Arguments {unsupported} are not supported.'
        )

        super().__init__(self.message)


class MissingSpecializationError(Exception):
    """Exception raised when a specialized JitKernel was called with arguments
     that do not match any of the specialized versions of the JitKernel.

    Args:
        kernel_name (str): Name of kernel where the error was raised.
        sig: Unsupported argument types used to call a specialized JitKernel.
    """

    def __init__(self, kernel_name, argtypes) -> None:
        self.message = (
            f"No specialized version of the kernel {kernel_name} "
            f"exists for argument types: {argtypes}."
        )

        super().__init__(self.message)


class UnsupportedParforError(Exception):
    """Exception raised when a parfor node could not be lowered by Numba-dpex"""

    def __init__(self, extra_msg=None) -> None:
        self.message = "Expression cannot be offloaded"
        if extra_msg:
            self.message += " due to " + extra_msg
        super().__init__(self.message)


class IllegalIntEnumLiteralValueError(Exception):
    """Exception raised when an IntEnumLiteral is attempted to be created from
    a non FlagEnum attribute.
    """

    def __init__(self) -> None:
        self.message = (
            "An IntEnumLiteral can only be initialized from a FlagEnum member"
        )
        super().__init__(self.message)


class InternalError(Exception):
    """
    Exception raise when something in the internal compiler machinery failed.
    """

    def __init__(self, extra_msg=None) -> None:
        if extra_msg is None:
            self.message = "An internal compiler error happened and the "
            "compilation failed."
        else:
            self.message = extra_msg
        super().__init__(self.message)
