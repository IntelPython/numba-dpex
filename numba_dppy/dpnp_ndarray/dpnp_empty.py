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


import builtins

import dpnp
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic, overload, overload_classmethod

from .types import dpnp_ndarray_Type


@overload(dpnp.empty)
def _ol_dpnp_empty(n, usm_type, sycl_queue):
    usm_type_num = {
        "shared": 0,
        "device": 1,
        "host": 2,
    }[usm_type.literal_value]

    def impl(n, usm_type, sycl_queue):
        return dpnp_ndarray_Type._allocate(n, 7, usm_type_num, sycl_queue)

    return impl


@overload_classmethod(dpnp_ndarray_Type, "_allocate")
def _ol_dpnp_array_allocate(cls, allocsize, align, usm_type, sycl_queue):
    def impl(cls, allocsize, align, usm_type, sycl_queue):
        return intrin_alloc(allocsize, align, usm_type, sycl_queue)

    return impl


@intrinsic
def intrin_alloc(typingctx, allocsize, align, usm_type, sycl_queue):
    """Intrinsic to call into the allocator for Array"""
    from numba.core.base import BaseContext
    from numba.core.runtime.context import NRTContext
    from numba.core.typing.templates import Signature

    def MemInfo_new(context: NRTContext, builder, size, usm_type, queue):
        context._require_nrt()

        mod = builder.module
        fnargs = [cgutils.intp_t, cgutils.intp_t, cgutils.voidptr_t]
        fnty = ir.FunctionType(cgutils.voidptr_t, fnargs)
        fn = cgutils.get_or_insert_function(mod, fnty, "DPRT_MemInfo_new")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size, usm_type, queue])

    def codegen(context: BaseContext, builder, signature: Signature, args):
        [allocsize, align, usm_type, sycl_queue] = args
        meminfo = MemInfo_new(context.nrt, builder, allocsize, usm_type, sycl_queue)
        meminfo.name = "allocate_UsmArray"
        return meminfo

    from numba.core.typing import signature

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, align, usm_type, sycl_queue)
    return sig, codegen
