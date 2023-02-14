# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.errors import NumbaAssertionError
from numba.core.typing.arraydecl import _expand_integer
from numba.core.typing.templates import (
    AttributeTemplate,
    bound_function,
    infer_getattr,
    signature,
)

import numba_dpex
from numba_dpex.core.types import DpnpNdArray


@infer_getattr
class DpnpTemplate(AttributeTemplate):
    key = types.Module(numba_dpex)

    def resolve_dpnp(self, mod):
        return types.Module(numba_dpex.dpnp)


"""
This adds a shapeptr attribute to Numba type representing np.ndarray.
This allows us to get the raw pointer to the structure where the shape
of an ndarray is stored from an overloaded implementation
"""


@infer_getattr
class ArrayAttribute(AttributeTemplate):
    key = types.Array

    def resolve_shapeptr(self, ary):
        return types.voidptr


@infer_getattr
class ListAttribute(AttributeTemplate):
    key = types.List

    def resolve_size(self, ary):
        return types.int64

    def resolve_itemsize(self, ary):
        return types.int64

    def resolve_ctypes(self, ary):
        return types.voidptr


def generic_expand_cumulative(ary, args, kws):
    if args:
        raise NumbaAssertionError("args unsupported")
    if kws:
        raise NumbaAssertionError("kwargs unsupported")
    assert isinstance(ary, types.Array)

    if isinstance(ary, DpnpNdArray):
        return_type = DpnpNdArray(
            dtype=_expand_integer(ary.dtype),
            ndim=1,
            layout="C",
            usm_type=ary.usm_type,
            device=ary.device,
            queue=ary.queue,
            addrspace=ary.addrspace,
        )
        return signature(return_type, recvr=ary)

    return_type = types.Array(
        dtype=_expand_integer(ary.dtype), ndim=1, layout="C"
    )
    return signature(return_type, recvr=ary)


@infer_getattr
class DPNPArrayAttribute(ArrayAttribute):
    key = DpnpNdArray

    @bound_function("array.cumsum")
    def resolve_cumsum(self, ary, args, kws):
        return generic_expand_cumulative(ary, args, kws)
