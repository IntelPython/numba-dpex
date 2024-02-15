# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements the SPIR-V overloads for the kernel_api.AtomicRef class methods.
"""

from llvmlite import ir as llvmir
from numba import errors
from numba.core import cgutils, types
from numba.extending import intrinsic, overload, overload_method

from numba_dpex.core import itanium_mangler as ext_itanium_mangler
from numba_dpex.core.targets.kernel_target import CC_SPIR_FUNC, LLVM_SPIRV_ARGS
from numba_dpex.core.types import USMNdArray
from numba_dpex.experimental.flag_enum import FlagEnum
from numba_dpex.kernel_api import (
    AddressSpace,
    AtomicRef,
    MemoryOrder,
    MemoryScope,
)

from ..dpcpp_types import AtomicRefType
from ..target import DPEX_KERNEL_EXP_TARGET_NAME
from ._spv_atomic_inst_helper import (
    get_atomic_inst_name,
    get_memory_semantics_mask,
    get_scope,
)
from .spv_fn_generator import (
    get_or_insert_atomic_load_fn,
    get_or_insert_spv_atomic_exchange_fn,
    get_or_insert_spv_atomic_store_fn,
)


def _parse_enum_or_int_literal_(literal_int) -> int:
    """Parse an instance of an enum class or numba.core.types.Literal to its
    actual int value.

    Returns the Python int value for the numba Literal type.

    Args:
        literal_int: Instance of IntegerLiteral wrapping a Python int scalar
        value
    Raises:
        TypingError: If the literal_int is not an IntegerLiteral type.
    Returns:
        int: The Python int value extracted from the literal_int.
    """

    if isinstance(literal_int, types.IntegerLiteral):
        return literal_int.literal_value

    if isinstance(literal_int, FlagEnum):
        return literal_int.value

    raise errors.TypingError("Could not parse input as an IntegerLiteral")


def _intrinsic_helper(
    ty_context, ty_atomic_ref, ty_val, op_str  # pylint: disable=unused-argument
):
    sig = types.void(ty_atomic_ref, ty_val)

    def gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_ref_dtype = atomic_ref_ty.dtype
        retty = context.get_value_type(atomic_ref_dtype)

        data_attr_pos = context.data_model_manager.lookup(
            atomic_ref_ty
        ).get_field_position("ref")

        # TODO: evaluating the llvm-spirv flags that dpcpp uses
        context.extra_compile_options[LLVM_SPIRV_ARGS] = [
            "--spirv-ext=+SPV_EXT_shader_atomic_float_add"
        ]

        context.extra_compile_options[LLVM_SPIRV_ARGS] = [
            "--spirv-ext=+SPV_EXT_shader_atomic_float_min_max"
        ]

        ptr_type = retty.as_pointer()
        ptr_type.addrspace = atomic_ref_ty.address_space

        spirv_fn_arg_types = [
            ptr_type,
            llvmir.IntType(32),
            llvmir.IntType(32),
            retty,
        ]

        mangled_fn_name = ext_itanium_mangler.mangle_ext(
            get_atomic_inst_name(op_str, atomic_ref_dtype),
            [
                types.CPointer(
                    atomic_ref_dtype, addrspace=atomic_ref_ty.address_space
                ),
                "__spv.Scope.Flag",
                "__spv.MemorySemanticsMask.Flag",
                atomic_ref_dtype,
            ],
        )
        func = cgutils.get_or_insert_function(
            builder.module,
            llvmir.FunctionType(retty, spirv_fn_arg_types),
            mangled_fn_name,
        )
        func.calling_convention = CC_SPIR_FUNC
        spirv_memory_semantics_mask = get_memory_semantics_mask(
            atomic_ref_ty.memory_order
        )
        spirv_scope = get_scope(atomic_ref_ty.memory_scope)

        fn_args = [
            builder.extract_value(args[0], data_attr_pos),
            context.get_constant(types.int32, spirv_scope),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
            args[1],
        ]

        builder.call(func, fn_args)

    return sig, gen


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_fetch_add(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_add")


def _atomic_sub_float_wrapper(gen_fn):
    def gen(context, builder, sig, args):
        # args is a tuple, which is immutable
        # covert tuple to list obj first before replacing arg[1]
        # with fneg and convert back to tuple again.
        args_lst = list(args)
        args_lst[1] = builder.fneg(args[1])
        args = tuple(args_lst)

        gen_fn(context, builder, sig, args)

    return gen


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_fetch_sub(ty_context, ty_atomic_ref, ty_val):
    if ty_atomic_ref.dtype in (types.float32, types.float64):
        # dpcpp does not support ``__spirv_AtomicFSubEXT``. fetch_sub
        # for floats is implemented by negating the value and calling fetch_add.
        # For example, A.fetch_sub(A, val) is implemented as A.fetch_add(-val).
        sig, gen = _intrinsic_helper(
            ty_context, ty_atomic_ref, ty_val, "fetch_add"
        )
        return sig, _atomic_sub_float_wrapper(gen)

    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_sub")


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_fetch_min(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_min")


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_fetch_max(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_max")


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_fetch_and(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_and")


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_fetch_or(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_or")


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_fetch_xor(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_xor")


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_atomic_ref_ctor(
    ty_context, ref, ty_index, ty_retty_ref  # pylint: disable=unused-argument
):
    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(ref, ty_index, ty_retty_ref)

    def codegen(context, builder, sig, args):
        ref = args[0]
        index_pos = args[1]

        dmm = context.data_model_manager
        data_attr_pos = dmm.lookup(sig.args[0]).get_field_position("data")
        data_attr = builder.extract_value(ref, data_attr_pos)

        with builder.goto_entry_block():
            ptr_to_data_attr = builder.alloca(data_attr.type)
        builder.store(data_attr, ptr_to_data_attr)
        ref_ptr_value = builder.gep(builder.load(ptr_to_data_attr), [index_pos])

        atomic_ref_struct = cgutils.create_struct_proxy(ty_retty)(
            context, builder
        )
        ref_attr_pos = dmm.lookup(ty_retty).get_field_position("ref")
        atomic_ref_struct[ref_attr_pos] = ref_ptr_value
        # pylint: disable=protected-access
        return atomic_ref_struct._getvalue()

    return (
        sig,
        codegen,
    )


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_load(
    ty_context, ty_atomic_ref  # pylint: disable=unused-argument
):
    sig = ty_atomic_ref.dtype(ty_atomic_ref)

    def _intrinsic_load_gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        fn = get_or_insert_atomic_load_fn(
            context, builder.module, atomic_ref_ty
        )

        spirv_memory_semantics_mask = get_memory_semantics_mask(
            atomic_ref_ty.memory_order
        )
        spirv_scope = get_scope(atomic_ref_ty.memory_scope)

        fn_args = [
            builder.extract_value(
                args[0],
                context.data_model_manager.lookup(
                    atomic_ref_ty
                ).get_field_position("ref"),
            ),
            context.get_constant(types.int32, spirv_scope),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
        ]

        return builder.call(fn, fn_args)

    return sig, _intrinsic_load_gen


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_store(
    ty_context, ty_atomic_ref, ty_val
):  # pylint: disable=unused-argument
    sig = types.void(ty_atomic_ref, ty_val)

    def _intrinsic_store_gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_store_fn = get_or_insert_spv_atomic_store_fn(
            context, builder.module, atomic_ref_ty
        )

        atomic_store_fn_args = [
            builder.extract_value(
                args[0],
                context.data_model_manager.lookup(
                    atomic_ref_ty
                ).get_field_position("ref"),
            ),
            context.get_constant(
                types.int32, get_scope(atomic_ref_ty.memory_scope)
            ),
            context.get_constant(
                types.int32,
                get_memory_semantics_mask(atomic_ref_ty.memory_order),
            ),
            args[1],
        ]

        builder.call(atomic_store_fn, atomic_store_fn_args)

    return sig, _intrinsic_store_gen


@intrinsic(target=DPEX_KERNEL_EXP_TARGET_NAME)
def _intrinsic_exchange(
    ty_context, ty_atomic_ref, ty_val  # pylint: disable=unused-argument
):
    sig = ty_atomic_ref.dtype(ty_atomic_ref, ty_val)

    def _intrinsic_exchange_gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_exchange_fn = get_or_insert_spv_atomic_exchange_fn(
            context, builder.module, atomic_ref_ty
        )

        atomic_exchange_fn_args = [
            builder.extract_value(
                args[0],
                context.data_model_manager.lookup(
                    atomic_ref_ty
                ).get_field_position("ref"),
            ),
            context.get_constant(
                types.int32, get_scope(atomic_ref_ty.memory_scope)
            ),
            context.get_constant(
                types.int32,
                get_memory_semantics_mask(atomic_ref_ty.memory_order),
            ),
            args[1],
        ]

        return builder.call(atomic_exchange_fn, atomic_exchange_fn_args)

    return sig, _intrinsic_exchange_gen


def _check_if_supported_ref(ref):
    supported = True

    if not isinstance(ref, USMNdArray):
        raise errors.TypingError(
            f"Cannot create an AtomicRef from {ref}. "
            "An AtomicRef can only be constructed from a 0-dimensional "
            "dpctl.tensor.usm_ndarray or a dpnp.ndarray array."
        )
    if ref.dtype not in [
        types.int32,
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
    if ref.dtype in [types.int64, types.float64, types.uint64]:
        if not ref.queue.device_has_aspect_atomic64:
            raise errors.TypingError(
                "Targeted device does not support 64-bit atomic operations."
            )

    return supported


@overload(
    AtomicRef,
    prefer_literal=True,
    target=DPEX_KERNEL_EXP_TARGET_NAME,
)
def ol_atomic_ref(
    ref,
    index=0,
    memory_order=MemoryOrder.RELAXED,
    memory_scope=MemoryScope.DEVICE,
    address_space=AddressSpace.GLOBAL,
):
    """Overload of the constructor for the class
    class:`numba_dpex.kernel_api.AtomicRef`.

    Raises:
        errors.TypingError: If the `ref` argument is not a UsmNdArray type.
        errors.TypingError: If the dtype of the `ref` is not supported in an
        AtomicRef.
        errors.TypingError: If the device does not support atomic operations on
        the dtype of the `ref`.
        errors.TypingError: If the `memory_order`, `address_type`, or
        `memory_scope` arguments could not be parsed as integer literals.
        errors.TypingError: If the `address_space` argument is different from
        the address space attribute of the `ref` argument.
        errors.TypingError: If the address space is PRIVATE.

    """
    _check_if_supported_ref(ref)

    try:
        _address_space = _parse_enum_or_int_literal_(address_space)
    except errors.TypingError as exc:
        raise errors.TypingError(
            "Address space argument to AtomicRef constructor should "
            "be an IntegerLiteral."
        ) from exc

    try:
        _memory_order = _parse_enum_or_int_literal_(memory_order)
    except errors.TypingError as exc:
        raise errors.TypingError(
            "Memory order argument to AtomicRef constructor should "
            "be an IntegerLiteral."
        ) from exc

    try:
        _memory_scope = _parse_enum_or_int_literal_(memory_scope)
    except errors.TypingError as exc:
        raise errors.TypingError(
            "Memory scope argument to AtomicRef constructor should "
            "be an IntegerLiteral."
        ) from exc

    if _address_space != ref.addrspace:
        raise errors.TypingError(
            "The address_space specified via the AtomicRef constructor "
            f"{_address_space} does not match the address space "
            f"{ref.addrspace} of the referred object for which the AtomicRef "
            "is to be constructed."
        )

    if _address_space == AddressSpace.PRIVATE:
        raise errors.TypingError(
            f"Unsupported address space value {_address_space}. Supported "
            f"address space values are: Generic ({AddressSpace.GENERIC}), "
            f"Global ({AddressSpace.GLOBAL}), and Local ({AddressSpace.LOCAL})."
        )

    ty_retty = AtomicRefType(
        dtype=ref.dtype,
        memory_order=_memory_order,
        memory_scope=_memory_scope,
        address_space=_address_space,
    )

    def ol_atomic_ref_ctor_impl(
        ref,
        index=0,
        memory_order=MemoryOrder.RELAXED,  # pylint: disable=unused-argument
        memory_scope=MemoryScope.DEVICE,  # pylint: disable=unused-argument
        address_space=AddressSpace.GLOBAL,  # pylint: disable=unused-argument
    ):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_atomic_ref_ctor(ref, index, ty_retty)

    return ol_atomic_ref_ctor_impl


@overload_method(AtomicRefType, "fetch_add", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_fetch_add(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.fetch_add`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::fetch_add` function.

    Raises:
        TypingError: When the dtype of the aggregator value does not match the
        dtype of the AtomicRef type.
    """
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to add: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_fetch_add_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_fetch_add(atomic_ref, val)

    return ol_fetch_add_impl


@overload_method(AtomicRefType, "fetch_sub", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_fetch_sub(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.fetch_sub`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::fetch_sub` function.

    Raises:
        TypingError: When the dtype of the aggregator value does not match the
        dtype of the AtomicRef type.
    """
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to sub: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_fetch_sub_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_fetch_sub(atomic_ref, val)

    return ol_fetch_sub_impl


@overload_method(AtomicRefType, "fetch_min", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_fetch_min(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.fetch_min`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::fetch_min` function.

    Raises:
        TypingError: When the dtype of the aggregator value does not match the
        dtype of the AtomicRef type.
    """
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to find min: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_fetch_min_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_fetch_min(atomic_ref, val)

    return ol_fetch_min_impl


@overload_method(AtomicRefType, "fetch_max", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_fetch_max(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.fetch_max`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::fetch_max` function.

    Raises:
        TypingError: When the dtype of the aggregator value does not match the
        dtype of the AtomicRef type.
    """
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to find max: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_fetch_max_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_fetch_max(atomic_ref, val)

    return ol_fetch_max_impl


@overload_method(AtomicRefType, "fetch_and", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_fetch_and(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.fetch_and`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::fetch_and` function.

    Raises:
        TypingError: When the dtype of the aggregator value does not match the
        dtype of the AtomicRef type.
    """
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to and: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    if atomic_ref.dtype not in (types.int32, types.int64):
        raise errors.TypingError(
            "fetch_and operation only supported on int32 and int64 dtypes."
        )

    def ol_fetch_and_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_fetch_and(atomic_ref, val)

    return ol_fetch_and_impl


@overload_method(AtomicRefType, "fetch_or", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_fetch_or(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.fetch_or`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::fetch_or` function.

    Raises:
        TypingError: When the dtype of the aggregator value does not match the
        dtype of the AtomicRef type.
    """
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to or: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    if atomic_ref.dtype not in (types.int32, types.int64):
        raise errors.TypingError(
            "fetch_or operation only supported on int32 and int64 dtypes."
        )

    def ol_fetch_or_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_fetch_or(atomic_ref, val)

    return ol_fetch_or_impl


@overload_method(AtomicRefType, "fetch_xor", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_fetch_xor(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.fetch_xor`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::fetch_xor` function.

    Raises:
        TypingError: When the dtype of the aggregator value does not match the
        dtype of the AtomicRef type.
    """
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to xor: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    if atomic_ref.dtype not in (types.int32, types.int64):
        raise errors.TypingError(
            "fetch_xor operation only supported on int32 and int64 dtypes."
        )

    def ol_fetch_xor_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_fetch_xor(atomic_ref, val)

    return ol_fetch_xor_impl


@overload_method(AtomicRefType, "load", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_load(atomic_ref):  # pylint: disable=unused-argument
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.load`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::load` function.

    """

    def ol_load_impl(atomic_ref):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_load(atomic_ref)

    return ol_load_impl


@overload_method(AtomicRefType, "store", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_store(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.store`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::store` function.

    Raises:
        TypingError: When the dtype of the value stored does not match the
        dtype of the AtomicRef type.
    """

    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to store: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_store_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_store(atomic_ref, val)

    return ol_store_impl


@overload_method(AtomicRefType, "exchange", target=DPEX_KERNEL_EXP_TARGET_NAME)
def ol_exchange(atomic_ref, val):
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.exchange`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::exchange` function.

    Raises:
        TypingError: When the dtype of the value passed to `exchange`
        does not match the dtype of the AtomicRef type.
    """

    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to exchange: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_exchange_impl(atomic_ref, val):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_exchange(atomic_ref, val)

    return ol_exchange_impl
