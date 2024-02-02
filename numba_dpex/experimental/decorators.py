# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

""" The set of experimental decorators provided by numba_dpex that are not yet
ready to move to numba_dpex.core.
"""
import inspect
from warnings import warn

from numba.core import sigutils, typeinfer
from numba.core.target_extension import (
    jit_registry,
    resolve_dispatcher_from_str,
    target_registry,
)

from numba_dpex._kernel_api_impl.spirv.dispatcher import SPVKernelDispatcher
from numba_dpex.core.targets.kernel_target import CompilationMode

from .target import DPEX_KERNEL_EXP_TARGET_NAME


def _parse_func_or_sig(signature_or_function):
    # Handle signature (borrowed from numba). swapped signature and list check
    if signature_or_function is None:
        # No signature, no function
        pyfunc = None
        sigs = []
    elif sigutils.is_signature(signature_or_function):
        # A single signature is passed
        pyfunc = None
        sigs = [signature_or_function]
    elif isinstance(signature_or_function, list):
        # A list of signatures is passed
        pyfunc = None
        sigs = signature_or_function
    else:
        # A function is passed
        pyfunc = signature_or_function
        sigs = []

    return pyfunc, sigs


def kernel(func_or_sig=None, **options):
    """A decorator to define a kernel function.

    A kernel function is conceptually equivalent to a SYCL kernel function, and
    gets compiled into either an OpenCL or a LevelZero SPIR-V binary kernel.
    A kernel decorated Python function has the following restrictions:

        * The function can not return any value.
        * All array arguments passed to a kernel should adhere to compute
          follows data programming model.
    """

    # dispatcher is a type:
    # <class 'numba_dpex.experimental.kernel_dispatcher.KernelDispatcher'>
    dispatcher = resolve_dispatcher_from_str(DPEX_KERNEL_EXP_TARGET_NAME)
    if "_compilation_mode" in options:
        user_compilation_mode = options["_compilation_mode"]
        warn(
            "_compilation_mode is an internal flag that should not be set "
            "in the decorator. The decorator defined option "
            f"{user_compilation_mode} is going to be ignored."
        )
    options["_compilation_mode"] = CompilationMode.KERNEL

    # FIXME: The options need to be evaluated and checked here like it is
    # done in numba.core.decorators.jit

    func, sigs = _parse_func_or_sig(func_or_sig)
    for sig in sigs:
        if isinstance(sig, str):
            raise NotImplementedError(
                "Specifying signatures as string is not yet supported by numba-dpex"
            )

    def _kernel_dispatcher(pyfunc):
        disp: SPVKernelDispatcher = dispatcher(
            pyfunc=pyfunc,
            targetoptions=options,
        )

        if len(sigs) > 0:
            with typeinfer.register_dispatcher(disp):
                for sig in sigs:
                    disp.compile(sig)
                disp.disable_compile()

        return disp

    if func is None:
        return _kernel_dispatcher

    if not inspect.isfunction(func):
        raise ValueError(
            "Argument passed to the kernel decorator is neither a "
            "function object, nor a signature. If you are trying to "
            "specialize the kernel that takes a single argument, specify "
            "the return type as void explicitly."
        )
    return _kernel_dispatcher(func)


def device_func(func_or_sig=None, **options):
    """Generates a function with a device-only calling convention, e.g.,
    spir_func for SPIR-V based devices.

    The decorator is used to compile overloads in the DpexKernelTarget and
    users should use the decorator to define functions that are only callable
    from inside another device_func or a kernel.

    A device_func is not compiled down to device binary IR and instead left as
    LLVM IR. It is done so that the function can be inlined fully into the
    kernel module from where it is used at the LLVM level, leading to more
    optimization opportunities.

    Returns:
        KernelDispatcher: A KernelDispatcher instance with the
        _compilation_mode option set to DEVICE_FUNC.
    """
    dispatcher = resolve_dispatcher_from_str(DPEX_KERNEL_EXP_TARGET_NAME)

    if "_compilation_mode" in options:
        user_compilation_mode = options["_compilation_mode"]
        warn(
            "_compilation_mode is an internal flag that should not be set "
            "in the decorator. The decorator defined option "
            f"{user_compilation_mode} is going to be ignored."
        )
    options["_compilation_mode"] = CompilationMode.DEVICE_FUNC

    def _kernel_dispatcher(pyfunc):
        return dispatcher(
            pyfunc=pyfunc,
            targetoptions=options,
        )

    if func_or_sig is None:
        return _kernel_dispatcher

    return _kernel_dispatcher(func_or_sig)


jit_registry[target_registry[DPEX_KERNEL_EXP_TARGET_NAME]] = device_func
