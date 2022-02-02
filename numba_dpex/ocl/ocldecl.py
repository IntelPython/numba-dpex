# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.typing.npydecl import parse_dtype, parse_shape
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    CallableTemplate,
    ConcreteTemplate,
    Registry,
    signature,
)

import numba_dpex as dpex
from numba_dpex.core.types import Array
from numba_dpex.utils import address_space

registry = Registry()
intrinsic = registry.register
intrinsic_attr = registry.register_attr
intrinsic_global = registry.register_global

# register_number_classes(intrinsic_global)


def register(function, args):
    base = ConcreteTemplate
    name = f"Ocl_{function.__name__}"

    if isinstance(args, list):
        cases = [signature(*a) for a in args]
    else:
        cases = [signature(*args)]

    dct = {"key": function, "cases": cases}
    cls = type(base)(name, (base,), dct)

    globals()[name] = intrinsic(cls)
    intrinsic_global(function, types.Function(cls))


register(dpex.get_global_id, (types.intp, types.uint32))
register(dpex.get_local_id, (types.intp, types.uint32))
register(dpex.get_group_id, (types.intp, types.uint32))
register(dpex.get_num_groups, (types.intp, types.uint32))
register(dpex.get_work_dim, (types.intp, types.uint32))
register(dpex.get_global_size, (types.intp, types.uint32))
register(dpex.get_local_size, (types.intp, types.uint32))
register(dpex.barrier, [(types.intp, types.uint32), (types.void,)])
register(dpex.mem_fence, (types.intp, types.uint32))
register(dpex.sub_group_barrier, (types.void,))

# dpex.atomic submodule -------------------------------------------------------


@intrinsic
class Ocl_atomic_add(AbstractTemplate):
    key = dpex.atomic.add

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic
class Ocl_atomic_sub(AbstractTemplate):
    key = dpex.atomic.sub

    def generic(self, args, kws):
        assert not kws
        ary, idx, val = args

        if ary.ndim == 1:
            return signature(ary.dtype, ary, types.intp, ary.dtype)
        elif ary.ndim > 1:
            return signature(ary.dtype, ary, idx, ary.dtype)


@intrinsic_attr
class OclAtomicTemplate(AttributeTemplate):
    key = types.Module(dpex.atomic)

    def resolve_add(self, mod):
        return types.Function(Ocl_atomic_add)

    def resolve_sub(self, mod):
        return types.Function(Ocl_atomic_sub)


intrinsic_global(dpex.atomic.add, types.Function(Ocl_atomic_add))
intrinsic_global(dpex.atomic.sub, types.Function(Ocl_atomic_sub))

# dpex.local submodule -------------------------------------------------------


@intrinsic
class OCL_local_array(CallableTemplate):
    key = dpex.local.array

    def generic(self):
        def typer(shape, dtype):

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    return None
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any(
                    [not isinstance(s, types.IntegerLiteral) for s in shape]
                ):
                    return None
            else:
                return None

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return Array(
                    dtype=nb_dtype,
                    ndim=ndim,
                    layout="C",
                    addrspace=address_space.LOCAL,
                )

        return typer


@intrinsic_attr
class OclLocalTemplate(AttributeTemplate):
    key = types.Module(dpex.local)

    def resolve_array(self, mod):
        return types.Function(OCL_local_array)


# dpex.private submodule -------------------------------------------------------


@intrinsic
class OCL_private_array(CallableTemplate):
    key = dpex.private.array

    def generic(self):
        def typer(shape, dtype):

            # Only integer literals and tuples of integer literals are valid
            # shapes
            if isinstance(shape, types.Integer):
                if not isinstance(shape, types.IntegerLiteral):
                    return None
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if any(
                    [not isinstance(s, types.IntegerLiteral) for s in shape]
                ):
                    return None
            else:
                return None

            ndim = parse_shape(shape)
            nb_dtype = parse_dtype(dtype)
            if nb_dtype is not None and ndim is not None:
                return Array(
                    dtype=nb_dtype,
                    ndim=ndim,
                    layout="C",
                    addrspace=address_space.PRIVATE,
                )

        return typer


@intrinsic_attr
class OclPrivateTemplate(AttributeTemplate):
    key = types.Module(dpex.private)

    def resolve_array(self, mod):
        return types.Function(OCL_private_array)


# OpenCL module --------------------------------------------------------------


@intrinsic_attr
class OclModuleTemplate(AttributeTemplate):
    key = types.Module(dpex)

    def resolve_get_global_id(self, mod):
        return types.Function(globals()["Ocl_get_global_id"])

    def resolve_get_local_id(self, mod):
        return types.Function(globals()["Ocl_get_local_id"])

    def resolve_get_global_size(self, mod):
        return types.Function(globals()["Ocl_get_global_size"])

    def resolve_get_local_size(self, mod):
        return types.Function(globals()["Ocl_get_local_size"])

    def resolve_get_num_groups(self, mod):
        return types.Function(globals()["Ocl_get_num_groups"])

    def resolve_get_work_dim(self, mod):
        return types.Function(globals()["Ocl_get_work_dim"])

    def resolve_get_group_id(self, mod):
        return types.Function(globals()["Ocl_get_group_id"])

    def resolve_barrier(self, mod):
        return types.Function(globals()["Ocl_barrier"])

    def resolve_mem_fence(self, mod):
        return types.Function(globals()["Ocl_mem_fence"])

    def resolve_sub_group_barrier(self, mod):
        return types.Function(globals()["Ocl_sub_group_barrier"])

    def resolve_atomic(self, mod):
        return types.Module(dpex.atomic)

    def resolve_local(self, mod):
        return types.Module(dpex.local)

    def resolve_private(self, mod):
        return types.Module(dpex.private)


# intrinsic

intrinsic_global(dpex, types.Module(dpex))
