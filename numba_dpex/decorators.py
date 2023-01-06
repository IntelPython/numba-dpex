# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from warnings import warn

from numba.core import sigutils, types

from numba_dpex.core.kernel_interface.dispatcher import (
    JitKernel,
    get_ordered_arg_access_types,
)
from numba_dpex.core.kernel_interface.func import (
    compile_func,
    compile_func_template,
)
from numba_dpex.utils import npytypes_array_to_dpex_array


def kernel(
    func_or_sig=None,
    access_types=None,
    debug=None,
    enable_cache=True,
):
    """A decorator to define a kernel function.

    A kernel function is conceptually equivalent to a SYCL kernel function, and
    gets compiled into either an OpenCL or a LevelZero SPIR-V binary kernel.
    A kernel decorated Python function has the following restrictions:

        * The function can not return any value.
        * All array arguments passed to a kernel should adhere to compute
          follows data programming model.
    """

    def _kernel_dispatcher(pyfunc, sigs=None):
        ordered_arg_access_types = get_ordered_arg_access_types(
            pyfunc, access_types
        )
        return JitKernel(
            pyfunc=pyfunc,
            debug_flags=debug,
            array_access_specifiers=ordered_arg_access_types,
            enable_cache=enable_cache,
            specialization_sigs=sigs,
        )

    if func_or_sig is None:
        return _kernel_dispatcher
    elif not sigutils.is_signature(func_or_sig):
        func = func_or_sig
        return _kernel_dispatcher(func)
    else:
        # Specialized signatures can either be a single signature or a list.
        # In case only one signature is provided convert it to a list
        if not isinstance(func_or_sig, list):
            func_or_sig = [func_or_sig]

        def _specialized_kernel_dispatcher(pyfunc):
            ordered_arg_access_types = get_ordered_arg_access_types(
                pyfunc, access_types
            )
            return JitKernel(
                pyfunc=pyfunc,
                debug_flags=debug,
                array_access_specifiers=ordered_arg_access_types,
                enable_cache=enable_cache,
                specialization_sigs=func_or_sig,
            )

        return _specialized_kernel_dispatcher


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
