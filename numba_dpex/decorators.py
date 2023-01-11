# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import inspect

from numba.core import sigutils

from numba_dpex.core.kernel_interface.dispatcher import (
    JitKernel,
    get_ordered_arg_access_types,
)
from numba_dpex.core.kernel_interface.func import (
    compile_func,
    compile_func_template,
)


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
    elif isinstance(func_or_sig, str):
        raise NotImplementedError(
            "Specifying signatures as string is not yet supported by numba-dpex"
        )
    elif isinstance(func_or_sig, list) or sigutils.is_signature(func_or_sig):
        # String signatures are not supported as passing usm_ndarray type as
        # a string is not possible. Numba's sigutils relies on the type being
        # available in Numba's types.__dpct__ and dpex types are not registered
        # there yet.
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

        # TODO: This function is 99% same as _kernel_dispatcher()
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
    else:
        func = func_or_sig
        if not inspect.isfunction(func):
            raise ValueError(
                "Argument passed to the kernel decorator is neither a "
                "function object, nor a signature. If you are trying to "
                "specialize the kernel that takes a single argument, specify "
                "the return type as void explicitly."
            )
        return _kernel_dispatcher(func)


def func(func_or_sig=None, debug=None, enable_cache=True):
    """
    TODO: make similar to kernel version and when the signature is a list
        remove the caching for specialization in func
    """

    def _func_autojit(pyfunc, debug=None):
        return compile_func_template(pyfunc, debug=debug)

    def _func_autojit_wrapper(debug=None):
        return _func_autojit

    if func_or_sig is None:
        return _func_autojit_wrapper(debug=debug)
    elif isinstance(func_or_sig, str):
        raise NotImplementedError(
            "Specifying signatures as string is not yet supported by numba-dpex"
        )
    elif isinstance(func_or_sig, list) or sigutils.is_signature(func_or_sig):
        # String signatures are not supported as passing usm_ndarray type as
        # a string is not possible. Numba's sigutils relies on the type being
        # available in Numba's types.__dpct__ and dpex types are not registered
        # there yet.
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

        def _wrapped(pyfunc):
            return compile_func(pyfunc, func_or_sig, debug=debug)

        return _wrapped
    else:
        func = func_or_sig
        return _func_autojit(func, debug=debug)
