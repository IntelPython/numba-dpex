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
    get_sycl_queue,
)


def kernel(signature=None, access_types=None, debug=False, queue=None):
    """JIT compile a python function conforming using the DPPY backend.

    A kernel is equvalent to an OpenCL kernel function, and has the
    same restrictions as definined by SPIR_KERNEL calling convention.
    """
    if signature is None:
        return autojit(debug=False, access_types=access_types, queue=queue)
    elif not sigutils.is_signature(signature):
        func = signature
        return autojit(debug=False, access_types=access_types, queue=queue)(func)
    else:
        return _kernel_jit(signature, debug, access_types, queue=queue)


def autojit(debug=False, access_types=None, queue=None):
    def _kernel_autojit(pyfunc):
        q = get_sycl_queue(queue)
        ordered_arg_access_types = get_ordered_arg_access_types(pyfunc, access_types)
        return JitDPPYKernel(pyfunc, ordered_arg_access_types, q)

    return _kernel_autojit


def _kernel_jit(signature, debug, access_types, queue=None):
    argtypes, restype = sigutils.normalize_signature(signature)
    if restype is not None and restype != types.void:
        msg = "DPPY kernel must have void return type but got {restype}"
        raise TypeError(msg.format(restype=restype))

    def _wrapped(pyfunc):
        q = get_sycl_queue(queue)
        ordered_arg_access_types = get_ordered_arg_access_types(pyfunc, access_types)
        return compile_kernel(q, pyfunc, argtypes, ordered_arg_access_types, debug)

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
