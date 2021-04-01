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

from numba.core import sigutils, types
from .compiler import (
    compile_kernel,
    JitDPPYKernel,
    compile_dppy_func_template,
    compile_dppy_func,
    get_ordered_arg_access_types,
)
import dpctl


def kernel(signature=None, access_types=None, debug=False):
    """JIT compile a python function conforming using the DPPY backend.

    A kernel is equvalent to an OpenCL kernel function, and has the
    same restrictions as definined by SPIR_KERNEL calling convention.
    """
    if signature is None:
        return autojit(debug=False, access_types=access_types)
    elif not sigutils.is_signature(signature):
        func = signature
        return autojit(debug=False, access_types=access_types)(func)
    else:
        return _kernel_jit(signature, debug, access_types)


def autojit(debug=False, access_types=None):
    def _kernel_autojit(pyfunc):
        ordered_arg_access_types = get_ordered_arg_access_types(pyfunc, access_types)
        return JitDPPYKernel(pyfunc, debug, ordered_arg_access_types)

    return _kernel_autojit


def _kernel_jit(signature, debug, access_types):
    argtypes, restype = sigutils.normalize_signature(signature)

    if restype is not None and restype != types.void:
        msg = "DPPY kernel must have void return type but got {restype}"
        raise TypeError(msg.format(restype=restype))

    def _wrapped(pyfunc):
        current_queue = dpctl.get_current_queue()
        ordered_arg_access_types = get_ordered_arg_access_types(pyfunc, access_types)
        # We create an instance of JitDPPYKernel to make sure at call time
        # we are going through the caching mechanism.
        dppy_kernel = JitDPPYKernel(pyfunc, debug, ordered_arg_access_types)
        # This will make sure we are compiling eagerly.
        dppy_kernel.specialize(argtypes, current_queue)
        return dppy_kernel

    return _wrapped


def func(signature=None):
    if signature is None:
        return _func_autojit
    elif not sigutils.is_signature(signature):
        func = signature
        return _func_autojit(func)
    else:
        return _func_jit(signature)


def _func_jit(signature):
    argtypes, restype = sigutils.normalize_signature(signature)

    def _wrapped(pyfunc):
        return compile_dppy_func(pyfunc, restype, argtypes)

    return _wrapped


def _func_autojit(pyfunc):
    return compile_dppy_func_template(pyfunc)
