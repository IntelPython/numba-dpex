# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

""" The set of experimental decorators provided by numba_dpex that are not yet
ready to move to numba_dpex.core.
"""
import inspect

from numba.core import sigutils
from numba.core.target_extension import jit_registry, target_registry

from .kernel_dispatcher import KernelDispatcher


def kernel(func_or_sig=None, **options):
    """A decorator to define a kernel function.

    A kernel function is conceptually equivalent to a SYCL kernel function, and
    gets compiled into either an OpenCL or a LevelZero SPIR-V binary kernel.
    A kernel decorated Python function has the following restrictions:

        * The function can not return any value.
        * All array arguments passed to a kernel should adhere to compute
          follows data programming model.
    """
    # FIXME: The options need to be evaluated and checked here like it is
    # done in numba.core.decorators.jit
    options["generate_device_ir"] = True

    def _kernel_dispatcher(pyfunc):
        return KernelDispatcher(
            pyfunc=pyfunc,
            targetoptions=options,
        )

    if func_or_sig is None:
        return _kernel_dispatcher

    if isinstance(func_or_sig, str):
        raise NotImplementedError(
            "Specifying signatures as string is not yet supported by numba-dpex"
        )

    if isinstance(func_or_sig, list) or sigutils.is_signature(func_or_sig):
        # String signatures are not supported as passing usm_ndarray type as
        # a string is not possible. Numba's sigutils relies on the type being
        # available in Numba's `types.__dict__` and dpex types are not
        # registered there yet.
        if isinstance(func_or_sig, list):
            for sig in func_or_sig:
                if isinstance(sig, str):
                    raise NotImplementedError(
                        "Specifying signatures as string is not yet supported "
                        "by numba-dpex"
                    )
        # Specialized signatures can either be a single signature or a list.
        # In case only one signature is provided convert it to a list
        if not isinstance(func_or_sig, list):
            func_or_sig = [func_or_sig]

        def _specialized_kernel_dispatcher(pyfunc):
            return KernelDispatcher(
                pyfunc=pyfunc,
            )

        return _specialized_kernel_dispatcher
    func = func_or_sig
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
        generate_device_ir flag set to False.
    """
    options["generate_device_ir"] = False

    def _kernel_dispatcher(pyfunc):
        return KernelDispatcher(
            pyfunc=pyfunc,
            targetoptions=options,
        )

    if func_or_sig is None:
        return _kernel_dispatcher

    return _kernel_dispatcher(func_or_sig)


jit_registry[target_registry["dpex_kernel"]] = device_func
