# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.typing.templates import AttributeTemplate, infer_getattr

import numba_dpex


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
