# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
from numba.core import sigutils, types

from numba_dpex.core.exceptions import KernelHasReturnValueError
from numba_dpex.core.kernel_interface.dispatcher import (
    Dispatcher,
    get_ordered_arg_access_types,
)
from numba_dpex.utils import npytypes_array_to_dpex_array

from .compiler import JitKernel, compile_func, compile_func_template


def kernel(signature=None, access_types=None, debug=None, enable_cache=True):
    """The decorator to write a numba_dpex kernel function.

    A kernel function is conceptually equivalent to a SYCL kernel function, and
    gets compiled into either an OpenCL or a LevelZero SPIR-V binary kernel.
    A dpex kernel imposes the following restrictions:

        * A numba_dpex.kernel function can not return any value.
        * All array arguments passed to a kernel should be of the same type
          and have the same dtype.
    """
    if signature is None:
        return autojit(
            debug=debug, access_types=access_types, enable_cache=enable_cache
        )
    elif not sigutils.is_signature(signature):
        func = signature
        return autojit(
            debug=debug, access_types=access_types, enable_cache=enable_cache
        )(func)
    else:
        return _kernel_jit(
            signature, debug, access_types, enable_cache=enable_cache
        )


def autojit(debug=None, access_types=None, enable_cache=True):
    def _kernel_dispatcher(pyfunc):
        ordered_arg_access_types = get_ordered_arg_access_types(
            pyfunc, access_types
        )
        disp = Dispatcher(
            pyfunc=pyfunc,
            debug_flags=debug,
            array_access_specifiers=ordered_arg_access_types,
            enable_cache=enable_cache,
        )

        return disp

    return _kernel_dispatcher


def _kernel_jit(signature, debug, access_types, enable_cache=True):
    argtypes, rettype = sigutils.normalize_signature(signature)
    argtypes = tuple(
        [
            npytypes_array_to_dpex_array(ty)
            if isinstance(ty, types.npytypes.Array)
            else ty
            for ty in argtypes
        ]
    )

    def _wrapped(pyfunc):
        current_queue = dpctl.get_current_queue()
        ordered_arg_access_types = get_ordered_arg_access_types(
            pyfunc, access_types
        )
        # We create an instance of JitKernel to make sure at call time
        # we are going through the caching mechanism.
        kernel = JitKernel(pyfunc, debug, ordered_arg_access_types)
        if enable_cache:
            kernel.enable_cache()

        # This will make sure we are compiling eagerly.
        kernel.specialize(argtypes, current_queue)
        return kernel

    return _wrapped


def func(signature=None, debug=None):
    if signature is None:
        return _func_autojit_wrapper(debug=debug)
    elif not sigutils.is_signature(signature):
        func = signature
        return _func_autojit(func, debug=debug)
    else:
        return _func_jit(signature, debug=debug)


def _func_jit(signature, debug=None):
    argtypes, restype = sigutils.normalize_signature(signature)
    argtypes = tuple(
        [
            npytypes_array_to_dpex_array(ty)
            if isinstance(ty, types.npytypes.Array)
            else ty
            for ty in argtypes
        ]
    )

    def _wrapped(pyfunc):
        return compile_func(pyfunc, restype, argtypes, debug=debug)

    return _wrapped


def _func_autojit_wrapper(debug=None):
    def _func_autojit(pyfunc, debug=debug):
        return compile_func_template(pyfunc, debug=debug)

    return _func_autojit


def _func_autojit(pyfunc, debug=None):
    return compile_func_template(pyfunc, debug=debug)
