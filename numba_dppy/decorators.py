# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dpctl
from numba.core import sigutils, types

from numba_dppy.utils import assert_no_return, npytypes_array_to_dpex_array

from .compiler import (
    JitKernel,
    compile_func,
    compile_func_template,
    get_ordered_arg_access_types,
)


def kernel(signature=None, access_types=None, debug=None):
    """JIT compile a python function conforming using the Dpex backend.

    A kernel is equivalent to an OpenCL kernel function, and has the
    same restrictions as defined by SPIR_KERNEL calling convention.
    """
    if signature is None:
        return autojit(debug=debug, access_types=access_types)
    elif not sigutils.is_signature(signature):
        func = signature
        return autojit(debug=debug, access_types=access_types)(func)
    else:
        return _kernel_jit(signature, debug, access_types)


def autojit(debug=None, access_types=None):
    def _kernel_autojit(pyfunc):
        ordered_arg_access_types = get_ordered_arg_access_types(
            pyfunc, access_types
        )
        return JitKernel(pyfunc, debug, ordered_arg_access_types)

    return _kernel_autojit


def _kernel_jit(signature, debug, access_types):
    argtypes, rettype = sigutils.normalize_signature(signature)
    argtypes = tuple(
        [
            npytypes_array_to_dpex_array(ty)
            if isinstance(ty, types.npytypes.Array)
            else ty
            for ty in argtypes
        ]
    )

    # Raises TypeError when users return anything inside @numba_dpex.kernel.
    assert_no_return(rettype)

    def _wrapped(pyfunc):
        current_queue = dpctl.get_current_queue()
        ordered_arg_access_types = get_ordered_arg_access_types(
            pyfunc, access_types
        )
        # We create an instance of JitKernel to make sure at call time
        # we are going through the caching mechanism.
        kernel = JitKernel(pyfunc, debug, ordered_arg_access_types)
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
