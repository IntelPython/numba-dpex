# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import operator
from functools import reduce

import dpctl
from llvmlite import binding as ll
from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.core.imputils import Registry
from numba.core.typing.npydecl import parse_dtype
from numba.np.arrayobj import get_itemsize

from numba_dpex import spirv_kernel_target
from numba_dpex.core import config
from numba_dpex.core.types import Array
from numba_dpex.kernel_api import AddressSpace as address_space
from numba_dpex.kernel_api_impl.spirv.codegen import SPIR_DATA_LAYOUT
from numba_dpex.ocl.atomics import atomic_helper

from . import stubs
from ._declare_function import _declare_function

registry = Registry()
lower = registry.lower

_void_value = llvmir.Constant(llvmir.IntType(8).as_pointer(), None)


# -----------------------------------------------------------------------------


@lower(stubs.get_global_id, types.uint32)
def get_global_id_impl(context, builder, sig, args):
    [dim] = args
    get_global_id = _declare_function(
        context, builder, "get_global_id", sig, ["unsigned int"]
    )
    res = builder.call(get_global_id, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_local_id, types.uint32)
def get_local_id_impl(context, builder, sig, args):
    [dim] = args
    get_local_id = _declare_function(
        context, builder, "get_local_id", sig, ["unsigned int"]
    )
    res = builder.call(get_local_id, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_group_id, types.uint32)
def get_group_id_impl(context, builder, sig, args):
    [dim] = args
    get_group_id = _declare_function(
        context, builder, "get_group_id", sig, ["unsigned int"]
    )
    res = builder.call(get_group_id, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_num_groups, types.uint32)
def get_num_groups_impl(context, builder, sig, args):
    [dim] = args
    get_num_groups = _declare_function(
        context, builder, "get_num_groups", sig, ["unsigned int"]
    )
    res = builder.call(get_num_groups, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_work_dim)
def get_work_dim_impl(context, builder, sig, args):
    get_work_dim = _declare_function(
        context, builder, "get_work_dim", sig, ["void"]
    )
    res = builder.call(get_work_dim, [])
    return res


@lower(stubs.get_global_size, types.uint32)
def get_global_size_impl(context, builder, sig, args):
    [dim] = args
    get_global_size = _declare_function(
        context, builder, "get_global_size", sig, ["unsigned int"]
    )
    res = builder.call(get_global_size, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.get_local_size, types.uint32)
def get_local_size_impl(context, builder, sig, args):
    [dim] = args
    get_local_size = _declare_function(
        context, builder, "get_local_size", sig, ["unsigned int"]
    )
    res = builder.call(get_local_size, [dim])
    return context.cast(builder, res, types.uintp, types.intp)


@lower(stubs.barrier, types.uint32)
def barrier_one_arg_impl(context, builder, sig, args):
    [flags] = args
    barrier = _declare_function(
        context, builder, "barrier", sig, ["unsigned int"]
    )
    builder.call(barrier, [flags])
    return _void_value


@lower(stubs.barrier)
def barrier_no_arg_impl(context, builder, sig, args):
    assert not args
    sig = types.void(types.uint32)
    barrier = _declare_function(
        context, builder, "barrier", sig, ["unsigned int"]
    )
    flags = context.get_constant(types.uint32, stubs.GLOBAL_MEM_FENCE)
    builder.call(barrier, [flags])
    return _void_value


@lower(stubs.mem_fence, types.uint32)
def mem_fence_impl(context, builder, sig, args):
    [flags] = args
    mem_fence = _declare_function(
        context, builder, "mem_fence", sig, ["unsigned int"]
    )
    builder.call(mem_fence, [flags])
    return _void_value


@lower(stubs.sub_group_barrier)
def sub_group_barrier_impl(context, builder, sig, args):
    assert not args
    sig = types.void(types.uint32)
    barrier = _declare_function(
        context, builder, "barrier", sig, ["unsigned int"]
    )
    flags = context.get_constant(types.uint32, stubs.LOCAL_MEM_FENCE)
    builder.call(barrier, [flags])
    return _void_value


def native_atomic_add(context, builder, sig, args):
    aryty, indty, valty = sig.args
    ary, inds, val = args
    dtype = aryty.dtype

    if indty == types.intp:
        indices = [inds]  # just a single integer
        indty = [indty]
    else:
        indices = cgutils.unpack_tuple(builder, inds, count=len(indty))
        indices = [
            context.cast(builder, i, t, types.intp)
            for t, i in zip(indty, indices)
        ]

    if dtype != valty:
        raise TypeError("expecting %s but got %s" % (dtype, valty))

    if aryty.ndim != len(indty):
        raise TypeError(
            "indexing %d-D array with %d-D index" % (aryty.ndim, len(indty))
        )

    lary = context.make_array(aryty)(context, builder, ary)
    ptr = cgutils.get_item_pointer(context, builder, aryty, lary, indices)

    if dtype == types.float32 or dtype == types.float64:
        context.extra_compile_options[spirv_kernel_target.LLVM_SPIRV_ARGS] = [
            "--spirv-ext=+SPV_EXT_shader_atomic_float_add"
        ]
        name = "__spirv_AtomicFAddEXT"
    elif dtype == types.int32 or dtype == types.int64:
        name = "__spirv_AtomicIAdd"
    else:
        raise TypeError("Unsupported type")

    assert name != ""

    ptr_type = context.get_value_type(dtype).as_pointer()
    ptr_type.addrspace = aryty.addrspace

    retty = context.get_value_type(sig.return_type)
    spirv_fn_arg_types = [
        ptr_type,
        llvmir.IntType(32),
        llvmir.IntType(32),
        context.get_value_type(sig.args[2]),
    ]

    from numba_dpex.core import itanium_mangler as ext_itanium_mangler

    numba_ptr_ty = types.CPointer(dtype, addrspace=ptr_type.addrspace)
    mangled_fn_name = ext_itanium_mangler.mangle_ext(
        name,
        [
            numba_ptr_ty,
            "__spv.Scope.Flag",
            "__spv.MemorySemanticsMask.Flag",
            valty,
        ],
    )

    fnty = llvmir.FunctionType(retty, spirv_fn_arg_types)
    fn = cgutils.get_or_insert_function(builder.module, fnty, mangled_fn_name)
    fn.calling_convention = spirv_kernel_target.CC_SPIR_FUNC

    sycl_memory_order = atomic_helper.sycl_memory_order.relaxed
    sycl_memory_scope = atomic_helper.sycl_memory_scope.device
    spirv_scope = atomic_helper.get_scope(sycl_memory_scope)
    spirv_memory_semantics_mask = atomic_helper.get_memory_semantics_mask(
        sycl_memory_order
    )
    fn_args = [
        ptr,
        context.get_constant(types.int32, spirv_scope),
        context.get_constant(types.int32, spirv_memory_semantics_mask),
        val,
    ]

    return builder.call(fn, fn_args)


def support_atomic(dtype: types.Type) -> bool:
    # This check should be the same as described in sycl documentation:
    # https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:atomic-references
    # If atomic is not supported, it will be emulated by the sycl compiler.
    return (
        dtype == types.int32
        or dtype == types.uint32
        or dtype == types.float32
        or dtype == types.int64
        or dtype == types.uint64
        or dtype == types.float64
    )


@lower(stubs.atomic.add, types.Array, types.intp, types.Any)
@lower(stubs.atomic.add, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.add, types.Array, types.Tuple, types.Any)
def atomic_add_tuple(context, builder, sig, args):
    dtype = sig.args[0].dtype
    if support_atomic(dtype):
        return native_atomic_add(context, builder, sig, args)
    else:
        raise TypeError(f"Atomic operation on unsupported type {dtype}")


def atomic_sub_wrapper(context, builder, sig, args):
    # dpcpp yet does not support ``__spirv_AtomicFSubEXT``. To support atomic.sub we
    # reuse atomic.add and negate the value. For example, atomic.add(A, index, -val) is
    # equivalent to atomic.sub(A, index, val).
    val = args[2]
    new_val = cgutils.alloca_once(
        builder,
        context.get_value_type(sig.args[2]),
        size=context.get_constant(types.uintp, 1),
        name="new_val_0",
    )
    val_dtype = sig.args[2]
    if val_dtype == types.float32 or val_dtype == types.float64:
        builder.store(
            builder.fmul(val, context.get_constant(sig.args[2], -1)), new_val
        )
    elif val_dtype == types.int32 or val_dtype == types.int64:
        builder.store(
            builder.mul(val, context.get_constant(sig.args[2], -1)), new_val
        )
    else:
        raise TypeError("Unsupported type %s" % val_dtype)

    args[2] = builder.load(new_val)

    return native_atomic_add(context, builder, sig, args)


@lower(stubs.atomic.sub, types.Array, types.intp, types.Any)
@lower(stubs.atomic.sub, types.Array, types.UniTuple, types.Any)
@lower(stubs.atomic.sub, types.Array, types.Tuple, types.Any)
def atomic_sub_tuple(context, builder, sig, args):
    dtype = sig.args[0].dtype
    if support_atomic(dtype):
        return atomic_sub_wrapper(context, builder, sig, args)
    else:
        raise TypeError(f"Atomic operation on unsupported type {dtype}")


@lower(stubs.private.array, types.IntegerLiteral, types.Any)
def dpex_private_array_integer(context, builder, sig, args):
    length = sig.args[0].literal_value
    dtype = parse_dtype(sig.args[1])
    return _generic_array(
        context,
        builder,
        shape=(length,),
        dtype=dtype,
        symbol_name="_dpex_pmem",
        addrspace=address_space.PRIVATE.value,
    )


@lower(stubs.private.array, types.Tuple, types.Any)
@lower(stubs.private.array, types.UniTuple, types.Any)
def dpex_private_array_tuple(context, builder, sig, args):
    shape = [s.literal_value for s in sig.args[0]]
    dtype = parse_dtype(sig.args[1])
    return _generic_array(
        context,
        builder,
        shape=shape,
        dtype=dtype,
        symbol_name="_dpex_pmem",
        addrspace=address_space.PRIVATE.value,
    )


@lower(stubs.local.array, types.IntegerLiteral, types.Any)
def dpex_local_array_integer(context, builder, sig, args):
    length = sig.args[0].literal_value
    dtype = parse_dtype(sig.args[1])
    return _generic_array(
        context,
        builder,
        shape=(length,),
        dtype=dtype,
        symbol_name="_dpex_lmem",
        addrspace=address_space.LOCAL.value,
    )


@lower(stubs.local.array, types.Tuple, types.Any)
@lower(stubs.local.array, types.UniTuple, types.Any)
def dpex_local_array_tuple(context, builder, sig, args):
    shape = [s.literal_value for s in sig.args[0]]
    dtype = parse_dtype(sig.args[1])
    return _generic_array(
        context,
        builder,
        shape=shape,
        dtype=dtype,
        symbol_name="_dpex_lmem",
        addrspace=address_space.LOCAL.value,
    )


def _generic_array(context, builder, shape, dtype, symbol_name, addrspace):
    """
    This function allows us to create generic arrays in different
    address spaces.
    """
    elemcount = reduce(operator.mul, shape)
    lldtype = context.get_data_type(dtype)
    laryty = llvmir.ArrayType(lldtype, elemcount)

    if addrspace == address_space.LOCAL.value:
        lmod = builder.module

        # Create global variable in the requested address-space
        gvmem = cgutils.add_global_variable(
            lmod, laryty, symbol_name, addrspace
        )

        if elemcount <= 0:
            raise ValueError("array length <= 0")
        else:
            gvmem.linkage = "internal"

        if dtype not in types.number_domain:
            raise TypeError("unsupported type: %s" % dtype)

    elif addrspace == address_space.PRIVATE.value:
        gvmem = cgutils.alloca_once(builder, laryty, name=symbol_name)
    else:
        raise NotImplementedError("addrspace {addrspace}".format(**locals()))

    # We need to add the addrspace to _make_array() function call as we want
    # the variable containing the reference of the memory to retain the
    # original address space of that memory. Before, we were casting the
    # memories allocated in local address space to global address space. This
    # approach does not let us identify the original address space of a memory
    # down the line.
    return _make_array(
        context, builder, gvmem, dtype, shape, addrspace=addrspace
    )


def _make_array(
    context,
    builder,
    dataptr,
    dtype,
    shape,
    layout="C",
    addrspace=address_space.GENERIC.value,
):
    ndim = len(shape)
    # Create array object
    aryty = Array(dtype=dtype, ndim=ndim, layout="C", addrspace=addrspace)
    ary = context.make_array(aryty)(context, builder)

    itemsize = get_itemsize(context, aryty)
    # Compute strides
    rstrides = [itemsize]
    for i, lastsize in enumerate(reversed(shape[1:])):
        rstrides.append(lastsize * rstrides[-1])
    strides = [s for s in reversed(rstrides)]

    kshape = [context.get_constant(types.intp, s) for s in shape]
    kstrides = [context.get_constant(types.intp, s) for s in strides]

    context.populate_array(
        ary,
        data=builder.bitcast(dataptr, ary.data.type),
        shape=cgutils.pack_array(builder, kshape),
        strides=cgutils.pack_array(builder, kstrides),
        itemsize=context.get_constant(types.intp, itemsize),
    )

    return ary._getvalue()
