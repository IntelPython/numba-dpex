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

from __future__ import print_function, division, absolute_import
from numba import types
from numba.core.typing.npydecl import register_number_classes, parse_dtype, parse_shape
from numba.core.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    AbstractTemplate,
    CallableTemplate,
    signature,
    Registry,
)
import numba_dppy, numba_dppy as dppy
from numba_dppy import target
from numba_dppy.dppy_array_type import DPPYArray

registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
intrinsic_global = registry.register_global

# register_number_classes(intrinsic_global)


@intrinsic
class Ocl_get_global_id(ConcreteTemplate):
    key = dppy.get_global_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_local_id(ConcreteTemplate):
    key = dppy.get_local_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_group_id(ConcreteTemplate):
    key = dppy.get_group_id
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_num_groups(ConcreteTemplate):
    key = dppy.get_num_groups
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_work_dim(ConcreteTemplate):
    key = dppy.get_work_dim
    cases = [signature(types.uint32)]


@intrinsic
class Ocl_get_global_size(ConcreteTemplate):
    key = dppy.get_global_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_get_local_size(ConcreteTemplate):
    key = dppy.get_local_size
    cases = [signature(types.intp, types.uint32)]


@intrinsic
class Ocl_barrier(ConcreteTemplate):
    key = dppy.barrier
    cases = [signature(types.void, types.uint32), signature(types.void)]


@intrinsic
class Ocl_mem_fence(ConcreteTemplate):
    key = dppy.mem_fence
    cases = [signature(types.void, types.uint32)]


@intrinsic
class Ocl_sub_group_barrier(ConcreteTemplate):
    key = dppy.sub_group_barrier

    cases = [signature(types.void)]


# dppy.atomic submodule -------------------------------------------------------


@intrinsic
class Ocl_atomic_add(AbstractTemplate):
    key = dppy.atomic.add

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic
class Ocl_atomic_sub(AbstractTemplate):
    key = dppy.atomic.sub

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic_attr
class OclAtomicTemplate(AttributeTemplate):
    key = types.Module(dppy.atomic)

    def resolve_add(self, mod):
        return types.Function(Ocl_atomic_add)

    def resolve_sub(self, mod):
        return types.Function(Ocl_atomic_sub)


intrinsic_global(dppy.atomic.add, types.Function(Ocl_atomic_add))
intrinsic_global(dppy.atomic.sub, types.Function(Ocl_atomic_sub))

# dppy.local submodule -------------------------------------------------------


@intrinsic
class OCL_local_array(CallableTemplate):
    key = dppy.local.array

    def generic(self):
        def typer(shape, dtype):

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    return None
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any([not isinstance(s, types.IntegerLiteral) for s in shape]):
                    return None
            else:
                return None

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return DPPYArray(
                    dtype=nb_dtype,
                    ndim=ndim,
                    layout="C",
                    addrspace=target.SPIR_LOCAL_ADDRSPACE,
                )

        return typer


@intrinsic_attr
class OclLocalTemplate(AttributeTemplate):
    key = types.Module(dppy.local)

    def resolve_array(self, mod):
        return types.Function(OCL_local_array)


# OpenCL module --------------------------------------------------------------


@intrinsic_attr
class OclModuleTemplate(AttributeTemplate):
    key = types.Module(dppy)

    def resolve_get_global_id(self, mod):
        return types.Function(Ocl_get_global_id)

    def resolve_get_local_id(self, mod):
        return types.Function(Ocl_get_local_id)

    def resolve_get_global_size(self, mod):
        return types.Function(Ocl_get_global_size)

    def resolve_get_local_size(self, mod):
        return types.Function(Ocl_get_local_size)

    def resolve_get_num_groups(self, mod):
        return types.Function(Ocl_get_num_groups)

    def resolve_get_work_dim(self, mod):
        return types.Function(Ocl_get_work_dim)

    def resolve_get_group_id(self, mod):
        return types.Function(Ocl_get_group_id)

    def resolve_barrier(self, mod):
        return types.Function(Ocl_barrier)

    def resolve_mem_fence(self, mod):
        return types.Function(Ocl_mem_fence)

    def resolve_sub_group_barrier(self, mod):
        return types.Function(Ocl_sub_group_barrier)

    def resolve_atomic(self, mod):
        return types.Module(dppy.atomic)

    def resolve_local(self, mod):
        return types.Module(dppy.local)


# intrinsic

# intrinsic_global(dppy, types.Module(dppy))
