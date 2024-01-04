# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import warnings

from numba.core import decorators, sigutils
from numba.core.target_extension import jit_registry, target_registry

from numba_dpex.core.kernel_interface.dispatcher import JitKernel
from numba_dpex.core.kernel_interface.func import (
    compile_func,
    compile_func_template,
)
from numba_dpex.core.pipelines.dpjit_compiler import get_compiler

from .config import USE_MLIR


def kernel(
    func_or_sig=None,
    debug=False,
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
        return JitKernel(
            pyfunc=pyfunc,
            debug_flags=debug,
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
        # available in Numba's `types.__dict__` and dpex types are not registered
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

        def _specialized_kernel_dispatcher(pyfunc):
            return JitKernel(
                pyfunc=pyfunc,
                debug_flags=debug,
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


def func(func_or_sig=None, debug=False, enable_cache=True):
    """A decorator to define a kernel device function.

    Device functions are functions that can be only invoked from a kernel
    and not from a host function. This provides a special decorator
    `numba_dpex.func` specifically to implement a device function.

    A device function can be invoked from another device function and
    unlike a kernel function, a device function can return a value like
    normal functions.
    """

    def _func_autojit(pyfunc):
        return compile_func_template(
            pyfunc, debug=debug, enable_cache=enable_cache
        )

    if func_or_sig is None:
        return _func_autojit
    elif isinstance(func_or_sig, str):
        raise NotImplementedError(
            "Specifying signatures as string is not yet supported by numba-dpex"
        )
    elif isinstance(func_or_sig, list) or sigutils.is_signature(func_or_sig):
        # String signatures are not supported as passing usm_ndarray type as
        # a string is not possible. Numba's sigutils relies on the type being
        # available in Numba's types.__dict__ and dpex types are not registered
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
        # no signature
        func = func_or_sig
        return _func_autojit(func)


# ----------------- Experimental dpjit decorator ------------------------------#


def dpjit(*args, **kws):
    if "nopython" in kws:
        warnings.warn(
            "nopython is set for dpjit and is ignored", RuntimeWarning
        )
    if "forceobj" in kws:
        warnings.warn(
            "forceobj is set for dpjit and is ignored", RuntimeWarning
        )
        del kws["forceobj"]
    if "pipeline_class" in kws:
        warnings.warn(
            "pipeline class is set for dpjit and is ignored", RuntimeWarning
        )
        del kws["pipeline_class"]

    use_mlir = kws.pop("use_mlir", bool(USE_MLIR))

    kws.update({"nopython": True})
    kws.update({"parallel": True})
    kws.update({"pipeline_class": get_compiler(use_mlir)})

    # FIXME: When trying to use dpex's target context, overloads do not work
    # properly. We will turn on dpex target once the issue is fixed.

    # kws.update({"_target": "dpex"}) # noqa: E800

    return decorators.jit(*args, **kws)


# add it to the decorator registry, this is so e.g. @overload can look up a
# JIT function to do the compilation work.
jit_registry[target_registry["dpex"]] = dpjit
