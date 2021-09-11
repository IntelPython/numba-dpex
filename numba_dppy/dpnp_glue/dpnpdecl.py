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

from numba import types
from numba.core.types.misc import RawPointer
from numba.core.typing.templates import AttributeTemplate, infer_getattr

import numba_dppy


@infer_getattr
class DppyDpnpTemplate(AttributeTemplate):
    key = types.Module(numba_dppy)

    def resolve_dpnp(self, mod):
        return types.Module(numba_dppy.dpnp)


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
