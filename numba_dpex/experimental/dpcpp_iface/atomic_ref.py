# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir as llvmir
from numba import errors
from numba.core import cgutils, types
from numba.extending import intrinsic, overload, overload_method

from numba_dpex.core import itanium_mangler as ext_itanium_mangler
from numba_dpex.core.datamodel.models import dpex_data_model_manager
from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.types import USMNdArray

from ..dpcpp_types import AtomicRefType
from ._spv_atomic_helper import get_memory_semantics_mask, get_scope


class AtomicRef(object):
    """The class provides the ability to perform atomic operations in a
    kernel function. The class is modeled after the ``sycl::atomic_ref`` class.

    """

    def __init__(self, ref, memory_order, memory_scope, address_space):
        self._memory_order = memory_order
        self._memory_scope = memory_scope
        self._address_space = address_space
        self._ref = ref

    def fetch_add(self, val):
        pass


@intrinsic
def _intrinsic_fetch_add(ty_context, ty_atomic_ref, ty_val):
    sig = types.void(ty_atomic_ref, ty_val)

    def gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_ref_dtype = atomic_ref_ty.dtype

        ref = args[0]
        data_attr_pos = dpex_data_model_manager.lookup(
            sig.args[0]
        ).get_field_position("ref")

        data_attr = builder.extract_value(ref, data_attr_pos)

        if atomic_ref_dtype in (types.float32, types.float64):
            # XXX Turn on once the overload is added to DpexKernelTarget
            # context.extra_compile_options[context.LLVM_SPIRV_ARGS] = [
            #     "--spirv-ext=+SPV_EXT_shader_atomic_float_add"
            # ]

            name = "__spirv_AtomicFAddEXT"
        elif atomic_ref_dtype in (types.int32, types.int64):
            name = "__spirv_AtomicIAdd"
        else:
            raise UnreachableError

        ptr_type = context.get_value_type(atomic_ref_dtype).as_pointer()
        ptr_type.addrspace = atomic_ref_ty.address_space.literal_value
        retty = context.get_value_type(atomic_ref_dtype)
        spirv_fn_arg_types = [
            ptr_type,
            llvmir.IntType(32),
            llvmir.IntType(32),
            context.get_value_type(atomic_ref_dtype),
        ]
        numba_ptr_ty = types.CPointer(
            atomic_ref_dtype, addrspace=ptr_type.addrspace
        )
        mangled_fn_name = ext_itanium_mangler.mangle_ext(
            name,
            [
                numba_ptr_ty,
                "__spv.Scope.Flag",
                "__spv.MemorySemanticsMask.Flag",
                atomic_ref_dtype,
            ],
        )
        fnty = llvmir.FunctionType(retty, spirv_fn_arg_types)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, mangled_fn_name
        )
        # XXX Change to DpexKernelTarget.SPIR_FUNC once overloads are added to
        # DpexKernelTarget
        fn.calling_convention = "spir_func"
        spirv_memory_semantics_mask = get_memory_semantics_mask(
            atomic_ref_ty.memory_order.literal_value
        )
        spirv_scope = get_scope(atomic_ref_ty.memory_scope.literal_value)

        # XXX Temporary address space cast is needed as we are using the
        # DpnpNdArrayModel used in dpjit. Once the overload is moved to
        # dpex_kernel we will not need the address space cast.
        addr_sp_casted_ref = builder.addrspacecast(data_attr, ptr_type)
        fn_args = [
            addr_sp_casted_ref,
            context.get_constant(types.int32, spirv_scope),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
            args[1],
        ]

        builder.call(fn, fn_args)

    return sig, gen


@intrinsic
def _intrinsic_atomic_ref_ctor(ty_context, ref, ty_retty_ref):
    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(ref, ty_retty_ref)

    def codegen(context, builder, sig, args):
        typ = sig.return_type
        ref = args[0]
        data_attr_pos = dpex_data_model_manager.lookup(
            sig.args[0]
        ).get_field_position("data")

        data_attr = builder.extract_value(ref, data_attr_pos)
        atomic_ref_struct = cgutils.create_struct_proxy(typ)(context, builder)

        # Populate the atomic ref data model
        ref_attr_pos = dpex_data_model_manager.lookup(
            ty_retty
        ).get_field_position("ref")
        builder.insert_value(
            atomic_ref_struct._getvalue(), data_attr, ref_attr_pos
        )
        return atomic_ref_struct._getvalue()

    return (
        sig,
        codegen,
    )


def _check_if_supported_ref(ref):
    supported = True

    if not (isinstance(ref, USMNdArray)):
        raise errors.TypingError(
            f"Cannot create an AtomicRef from {ref}. "
            "An AtomicRef can only be constructed from a 0-dimensional "
            "dpctl.tensor.usm_ndarray or a dpnp.ndarray array."
        )
    elif ref.ndim != 0:
        raise errors.TypingError(
            f"Cannot create an AtomicRef from a {ref.ndim}-dimensional tensor."
        )
    elif ref.dtype not in [
        types.uint32,
        types.float32,
        types.int64,
        types.uint64,
        types.float64,
    ]:
        raise errors.TypingError(
            "Cannot create an AtomicRef from a tensor with an unsupported "
            "element type."
        )
    elif ref.dtype in [types.int64, types.float64, types.uint64]:
        if not ref.queue.device_has_aspect_atomic64:
            raise errors.TypingError(
                "Targeted device does not support 64-bit atomic operations."
            )
    else:
        return supported


@overload(AtomicRef, prefer_literal=True, inline="always")
def ol_atomic_ref(ref, memory_order, memory_scope, address_space):
    _check_if_supported_ref(ref)

    ty_retty = AtomicRefType(
        dtype=ref.dtype,
        memory_order=memory_order,
        memory_scope=memory_scope,
        address_space=address_space,
        has_aspect_atomic64=ref.queue.device_has_aspect_atomic64,
    )

    def impl(ref, memory_order, memory_scope, address_space):
        return _intrinsic_atomic_ref_ctor(ref, ty_retty)

    return impl


@overload_method(AtomicRefType, "fetch_add", inline="always")
def ol_fetch_add(atomic_ref, val):
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to add: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def impl(atomic_ref, val):
        return _intrinsic_fetch_add(atomic_ref, val)

    return impl
