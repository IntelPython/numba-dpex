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
def _ol_dpnp_empty(n):
    def impl(n):
        return dpnp_ndarray_Type._allocate(n, 7)

    return impl


@overload_classmethod(dpnp_ndarray_Type, "_allocate")
def _ol_dpnp_array_allocate(cls, allocsize, align):
    def impl(cls, allocsize, align):
        return intrin_alloc(allocsize, align)

    return impl


@intrinsic
def intrin_alloc(typingctx, allocsize, align):
    """Intrinsic to call into the allocator for Array"""
    from numba.core.base import BaseContext
    from numba.core.runtime.context import NRTContext
    from numba.core.typing.templates import Signature

    def get_external_allocator(builder):
        """Get the Numba external allocator for USM memory."""
        fnty = ir.FunctionType(cgutils.voidptr_t, [])
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="usmarray_get_ext_allocator"
        )
        return builder.call(fn, [])

    def meminfo_alloc_aligned_external(
        context: NRTContext, builder, size, align, ext_allocator
    ):
        context._require_nrt()

        mod = builder.module
        u32 = ir.IntType(32)
        fnty = ir.FunctionType(
            cgutils.voidptr_t, [cgutils.intp_t, u32, cgutils.voidptr_t]
        )
        fn = cgutils.get_or_insert_function(
            mod, fnty, "NRT_MemInfo_alloc_safe_aligned_external"
        )
        fn.return_value.add_attribute("noalias")
        if isinstance(align, int):
            align = context._context.get_constant(types.uint32, align)
        else:
            assert align.type == u32, "align must be a uint32"
        return builder.call(fn, [size, align, ext_allocator])

    def codegen(context: BaseContext, builder, signature: Signature, args):
        ext_allocator = get_external_allocator(builder)

        [allocsize, align] = args
        align = cast_integer(
            context, builder, align, signature.args[1], types.uint32
        )
        meminfo = meminfo_alloc_aligned_external(
            context.nrt, builder, allocsize, align, ext_allocator
        )
        meminfo.name = "allocate_UsmArray"
        return meminfo

    from numba.core.typing import signature

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, align)
    return sig, codegen


def cast_integer(context, builder, val, fromty, toty):
    # XXX Shouldn't require this.
    if toty.bitwidth == fromty.bitwidth:
        # Just a change of signedness
        return val
    elif toty.bitwidth < fromty.bitwidth:
        # Downcast
        return builder.trunc(val, context.get_value_type(toty))
    elif fromty.signed:
        # Signed upcast
        return builder.sext(val, context.get_value_type(toty))
    else:
        # Unsigned upcast
        return builder.zext(val, context.get_value_type(toty))
