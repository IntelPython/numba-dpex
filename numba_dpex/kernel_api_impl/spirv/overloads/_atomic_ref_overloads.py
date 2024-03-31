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

from numba_dpex.core.types import USMNdArray
from numba_dpex.core.types.kernel_api.atomic_ref import AtomicRefType
from numba_dpex.core.utils import itanium_mangler as ext_itanium_mangler
from numba_dpex.kernel_api import (
    AddressSpace,
    AtomicRef,
    MemoryOrder,
    MemoryScope,
)
from numba_dpex.kernel_api.flag_enum import FlagEnum
from numba_dpex.kernel_api_impl.spirv.target import CC_SPIR_FUNC

from ..target import SPIRV_TARGET_NAME
from ._spv_atomic_inst_helper import (
    get_atomic_inst_name,
    get_memory_semantics_mask,
    get_scope,
)
from .spv_fn_declarations import (
    _SUPPORT_CONVERGENT,
    get_or_insert_atomic_load_fn,
    get_or_insert_spv_atomic_compare_exchange_fn,
    get_or_insert_spv_atomic_exchange_fn,
    get_or_insert_spv_atomic_store_fn,
)


def _normalize_indices(context, builder, indty, inds, aryty):
    """
    Convert integer indices into tuple of intp
    """
    if indty in types.integer_domain:
        indty = types.UniTuple(dtype=indty, count=1)
        indices = [inds]
    else:
        indices = cgutils.unpack_tuple(builder, inds, count=len(indty))
    indices = [
        context.cast(builder, i, t, types.intp) for t, i in zip(indty, indices)
    ]

    if aryty.ndim != len(indty):
        raise TypeError(
            f"indexing {aryty.ndim}-D array with {len(indty)}-D index"
        )

    return indty, indices


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
    sig = ty_atomic_ref.dtype(ty_atomic_ref, ty_val)

    def gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_ref_dtype = atomic_ref_ty.dtype
        retty = context.get_value_type(atomic_ref_dtype)

        data_attr_pos = context.data_model_manager.lookup(
            atomic_ref_ty
        ).get_field_position("ref")

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
        if _SUPPORT_CONVERGENT:
            func.attributes.add("convergent")
        func.attributes.add("nounwind")

        fn_args = [
            builder.extract_value(args[0], data_attr_pos),
            context.get_constant(
                types.int32, get_scope(atomic_ref_ty.memory_scope)
            ),
            context.get_constant(
                types.int32,
                get_memory_semantics_mask(atomic_ref_ty.memory_order),
            ),
            args[1],
        ]

        callinst = builder.call(func, fn_args)

        if _SUPPORT_CONVERGENT:
            callinst.attributes.add("convergent")
        callinst.attributes.add("nounwind")

        return callinst

    return sig, gen


@intrinsic(target=SPIRV_TARGET_NAME)
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

        return gen_fn(context, builder, sig, args)

    return gen


@intrinsic(target=SPIRV_TARGET_NAME)
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


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_fetch_min(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_min")


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_fetch_max(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_max")


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_fetch_and(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_and")


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_fetch_or(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_or")


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_fetch_xor(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_xor")


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_atomic_ref_ctor(
    ty_context, ref, ty_index, ty_retty_ref  # pylint: disable=unused-argument
):
    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(ref, ty_index, ty_retty_ref)

    def codegen(context, builder, sig, args):
        aryty, indty, _ = sig.args
        ary, inds, _ = args

        indty, indices = _normalize_indices(
            context, builder, indty, inds, aryty
        )

        lary = context.make_array(aryty)(context, builder, ary)
        ref_ptr_value = cgutils.get_item_pointer(
            context, builder, aryty, lary, indices, wraparound=True
        )

        atomic_ref_struct = cgutils.create_struct_proxy(ty_retty)(
            context, builder
        )
        atomic_ref_struct.ref = ref_ptr_value
        # pylint: disable=protected-access
        return atomic_ref_struct._getvalue()

    return (
        sig,
        codegen,
    )


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_load(
    ty_context, ty_atomic_ref  # pylint: disable=unused-argument
):
    sig = ty_atomic_ref.dtype(ty_atomic_ref)

    def _intrinsic_load_gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]

        atomic_ref_ptr = builder.extract_value(
            args[0],
            context.data_model_manager.lookup(atomic_ref_ty).get_field_position(
                "ref"
            ),
        )
        if sig.args[0].dtype == types.float32:
            atomic_ref_ptr = builder.bitcast(
                atomic_ref_ptr,
                llvmir.PointerType(
                    llvmir.IntType(32), addrspace=sig.args[0].address_space
                ),
            )
        elif sig.args[0].dtype == types.float64:
            atomic_ref_ptr = builder.bitcast(
                atomic_ref_ptr,
                llvmir.PointerType(
                    llvmir.IntType(64), addrspace=sig.args[0].address_space
                ),
            )

        fn_args = [
            atomic_ref_ptr,
            context.get_constant(
                types.int32, get_scope(atomic_ref_ty.memory_scope)
            ),
            context.get_constant(
                types.int32,
                get_memory_semantics_mask(atomic_ref_ty.memory_order),
            ),
        ]

        ret_val = builder.call(
            get_or_insert_atomic_load_fn(
                context, builder.module, atomic_ref_ty
            ),
            fn_args,
        )

        if _SUPPORT_CONVERGENT:
            ret_val.attributes.add("convergent")
        ret_val.attributes.add("nounwind")

        if sig.args[0].dtype == types.float32:
            ret_val = builder.bitcast(ret_val, llvmir.FloatType())
        elif sig.args[0].dtype == types.float64:
            ret_val = builder.bitcast(ret_val, llvmir.DoubleType())

        return ret_val

    return sig, _intrinsic_load_gen


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_store(
    ty_context, ty_atomic_ref, ty_val
):  # pylint: disable=unused-argument
    sig = types.void(ty_atomic_ref, ty_val)

    def _intrinsic_store_gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]

        store_arg = args[1]
        atomic_ref_ptr = builder.extract_value(
            args[0],
            context.data_model_manager.lookup(atomic_ref_ty).get_field_position(
                "ref"
            ),
        )
        if sig.args[0].dtype == types.float32:
            atomic_ref_ptr = builder.bitcast(
                atomic_ref_ptr,
                llvmir.PointerType(
                    llvmir.IntType(32), addrspace=sig.args[0].address_space
                ),
            )
            store_arg = builder.bitcast(store_arg, llvmir.IntType(32))
        elif sig.args[0].dtype == types.float64:
            atomic_ref_ptr = builder.bitcast(
                atomic_ref_ptr,
                llvmir.PointerType(
                    llvmir.IntType(64), addrspace=sig.args[0].address_space
                ),
            )
            store_arg = builder.bitcast(store_arg, llvmir.IntType(64))

        atomic_store_fn_args = [
            atomic_ref_ptr,
            context.get_constant(
                types.int32, get_scope(atomic_ref_ty.memory_scope)
            ),
            context.get_constant(
                types.int32,
                get_memory_semantics_mask(atomic_ref_ty.memory_order),
            ),
            store_arg,
        ]

        callinst = builder.call(
            get_or_insert_spv_atomic_store_fn(
                context, builder.module, atomic_ref_ty
            ),
            atomic_store_fn_args,
        )

        if _SUPPORT_CONVERGENT:
            callinst.attributes.add("convergent")
        callinst.attributes.add("nounwind")

    return sig, _intrinsic_store_gen


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_exchange(
    ty_context, ty_atomic_ref, ty_val  # pylint: disable=unused-argument
):
    sig = ty_atomic_ref.dtype(ty_atomic_ref, ty_val)

    def _intrinsic_exchange_gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]

        exchange_arg = args[1]
        atomic_ref_ptr = builder.extract_value(
            args[0],
            context.data_model_manager.lookup(atomic_ref_ty).get_field_position(
                "ref"
            ),
        )
        if sig.args[0].dtype == types.float32:
            atomic_ref_ptr = builder.bitcast(
                atomic_ref_ptr,
                llvmir.PointerType(
                    llvmir.IntType(32), addrspace=sig.args[0].address_space
                ),
            )
            exchange_arg = builder.bitcast(exchange_arg, llvmir.IntType(32))
        elif sig.args[0].dtype == types.float64:
            atomic_ref_ptr = builder.bitcast(
                atomic_ref_ptr,
                llvmir.PointerType(
                    llvmir.IntType(64), addrspace=sig.args[0].address_space
                ),
            )
            exchange_arg = builder.bitcast(exchange_arg, llvmir.IntType(64))

        atomic_exchange_fn_args = [
            atomic_ref_ptr,
            context.get_constant(
                types.int32, get_scope(atomic_ref_ty.memory_scope)
            ),
            context.get_constant(
                types.int32,
                get_memory_semantics_mask(atomic_ref_ty.memory_order),
            ),
            exchange_arg,
        ]

        ret_val = builder.call(
            get_or_insert_spv_atomic_exchange_fn(
                context, builder.module, atomic_ref_ty
            ),
            atomic_exchange_fn_args,
        )

        if _SUPPORT_CONVERGENT:
            ret_val.attributes.add("convergent")
        ret_val.attributes.add("nounwind")

        if sig.args[0].dtype == types.float32:
            ret_val = builder.bitcast(ret_val, llvmir.FloatType())
        elif sig.args[0].dtype == types.float64:
            ret_val = builder.bitcast(ret_val, llvmir.DoubleType())

        return ret_val

    return sig, _intrinsic_exchange_gen


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_compare_exchange(
    ty_context,  # pylint: disable=unused-argument
    ty_atomic_ref,
    ty_expected_ref,
    ty_desired,
    ty_expected_idx,
):
    sig = types.boolean(
        ty_atomic_ref, ty_expected_ref, ty_desired, ty_expected_idx
    )

    def _intrinsic_compare_exchange_gen(context, builder, sig, args):
        # get pointer to expected[expected_idx]
        data_attr = builder.extract_value(
            args[1],
            context.data_model_manager.lookup(sig.args[1]).get_field_position(
                "data"
            ),
        )
        with builder.goto_entry_block():
            ptr_to_data_attr = builder.alloca(data_attr.type)
        builder.store(data_attr, ptr_to_data_attr)
        expected_ref_ptr = builder.gep(
            builder.load(ptr_to_data_attr), [args[3]]
        )

        expected_arg = builder.load(expected_ref_ptr)
        desired_arg = args[2]
        atomic_ref_ptr = builder.extract_value(
            args[0],
            context.data_model_manager.lookup(sig.args[0]).get_field_position(
                "ref"
            ),
        )
        # add conditional bitcast for atomic_ref pointer,
        # expected[expected_idx], and desired
        if sig.args[0].dtype == types.float32:
            atomic_ref_ptr = builder.bitcast(
                atomic_ref_ptr,
                llvmir.PointerType(
                    llvmir.IntType(32), addrspace=sig.args[0].address_space
                ),
            )
            expected_arg = builder.bitcast(expected_arg, llvmir.IntType(32))
            desired_arg = builder.bitcast(desired_arg, llvmir.IntType(32))
        elif sig.args[0].dtype == types.float64:
            atomic_ref_ptr = builder.bitcast(
                atomic_ref_ptr,
                llvmir.PointerType(
                    llvmir.IntType(64), addrspace=sig.args[0].address_space
                ),
            )
            expected_arg = builder.bitcast(expected_arg, llvmir.IntType(64))
            desired_arg = builder.bitcast(desired_arg, llvmir.IntType(64))

        atomic_cmpexchg_fn_args = [
            atomic_ref_ptr,
            context.get_constant(
                types.int32, get_scope(sig.args[0].memory_scope)
            ),
            context.get_constant(
                types.int32,
                get_memory_semantics_mask(sig.args[0].memory_order),
            ),
            context.get_constant(
                types.int32,
                get_memory_semantics_mask(sig.args[0].memory_order),
            ),
            desired_arg,
            expected_arg,
        ]

        ret_val = builder.call(
            get_or_insert_spv_atomic_compare_exchange_fn(
                context, builder.module, sig.args[0]
            ),
            atomic_cmpexchg_fn_args,
        )

        if _SUPPORT_CONVERGENT:
            ret_val.attributes.add("convergent")
        ret_val.attributes.add("nounwind")

        # compare_exchange returns the old value stored in AtomicRef object.
        # If the return value is same as expected, then compare_exchange
        # succeeded in replacing AtomicRef object with desired.
        # If the return value is not same as expected, then store return
        # value in expected.
        # In either case, return result of cmp instruction.
        is_cmp_exchg_success = builder.icmp_signed("==", ret_val, expected_arg)

        with builder.if_else(is_cmp_exchg_success) as (then, otherwise):
            with then:
                pass
            with otherwise:
                if sig.args[0].dtype == types.float32:
                    ret_val = builder.bitcast(ret_val, llvmir.FloatType())
                elif sig.args[0].dtype == types.float64:
                    ret_val = builder.bitcast(ret_val, llvmir.DoubleType())
                builder.store(ret_val, expected_ref_ptr)
        return is_cmp_exchg_success

    return sig, _intrinsic_compare_exchange_gen


def _check_if_supported_ref(ref):
    supported = True

    if not isinstance(ref, USMNdArray):
        raise errors.TypingError(
            f"Cannot create an AtomicRef from {ref}. "
            "An AtomicRef can only be constructed from a 0-dimensional "
            "dpctl.tensor.usm_ndarray, dpnp.ndarray array, or a "
            "kernel_api.LocalAccessor object."
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
    target=SPIRV_TARGET_NAME,
)
def ol_atomic_ref(
    ref,
    index,
    memory_order=MemoryOrder.RELAXED,
    memory_scope=MemoryScope.DEVICE,
    address_space=None,
):
    """Overload of the constructor for the class
    class:`numba_dpex.kernel_api.AtomicRef`.

    Note that the ``address_space`` argument by default is set to None and is
    inferred from the address space of the ``ref`` argument. If an address space
    value is explicitly passed in, then it needs to match with the address space
    of the ``ref`` argument.

    TODO: The SYCL usage of the ``address_space`` argument to a sycl::atomic_ref
    constructor should be evaluated. Either we need to allow passing in a
    different address_space value w.r.t. the ``ref`` argument's address space
    and handle it the way SYCL does (probably by introducing an
    addresspace_cast), or the argument should be removed all together.

    Raises:
        errors.TypingError: If the `ref` argument is not a UsmNdArray or a
            LocalAccessorType type.
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

    if address_space is None:
        _address_space = ref.addrspace
    else:
        try:
            _address_space = _parse_enum_or_int_literal_(address_space)
        except errors.TypingError as exc:
            raise errors.TypingError(
                "Address space argument to AtomicRef constructor should "
                "be an IntegerLiteral."
            ) from exc
        if _address_space != ref.addrspace:
            raise errors.TypingError(
                "The address_space specified via the AtomicRef constructor "
                f"{_address_space} does not match the address space "
                f"{ref.addrspace} of the referred object for which the "
                "AtomicRef is to be constructed."
            )

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
        index,
        memory_order=MemoryOrder.RELAXED,  # pylint: disable=unused-argument
        memory_scope=MemoryScope.DEVICE,  # pylint: disable=unused-argument
        address_space=None,  # pylint: disable=unused-argument
    ):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_atomic_ref_ctor(ref, index, ty_retty)

    return ol_atomic_ref_ctor_impl


@overload_method(AtomicRefType, "fetch_add", target=SPIRV_TARGET_NAME)
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


@overload_method(AtomicRefType, "fetch_sub", target=SPIRV_TARGET_NAME)
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


@overload_method(AtomicRefType, "fetch_min", target=SPIRV_TARGET_NAME)
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


@overload_method(AtomicRefType, "fetch_max", target=SPIRV_TARGET_NAME)
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


@overload_method(AtomicRefType, "fetch_and", target=SPIRV_TARGET_NAME)
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


@overload_method(AtomicRefType, "fetch_or", target=SPIRV_TARGET_NAME)
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


@overload_method(AtomicRefType, "fetch_xor", target=SPIRV_TARGET_NAME)
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


@overload_method(AtomicRefType, "load", target=SPIRV_TARGET_NAME)
def ol_load(atomic_ref):  # pylint: disable=unused-argument
    """SPIR-V overload for :meth:`numba_dpex.kernel_api.AtomicRef.load`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::load` function.

    """

    def ol_load_impl(atomic_ref):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_load(atomic_ref)

    return ol_load_impl


@overload_method(AtomicRefType, "store", target=SPIRV_TARGET_NAME)
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


@overload_method(AtomicRefType, "exchange", target=SPIRV_TARGET_NAME)
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


@overload_method(
    AtomicRefType,
    "compare_exchange",
    target=SPIRV_TARGET_NAME,
)
def ol_compare_exchange(
    atomic_ref,
    expected_ref,
    desired,
    expected_idx=0,  # pylint: disable=unused-argument
):
    """SPIR-V overload for
    :meth:`numba_dpex.experimental.kernel_iface.AtomicRef.compare_exchange`.

    Generates the same LLVM IR instruction as dpcpp for the
    `atomic_ref::compare_exchange_strong` function.

    Raises:
        TypingError: When the dtype of the value passed to `compare_exchange`
        does not match the dtype of the AtomicRef type.
    """

    _check_if_supported_ref(expected_ref)

    if atomic_ref.dtype != expected_ref.dtype:
        raise errors.TypingError(
            f"Type of value to compare_exchange: {expected_ref} does not match the "
            f"type of the reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    if atomic_ref.dtype != desired:
        raise errors.TypingError(
            f"Type of value to compare_exchange: {desired} does not match the "
            f"type of the reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_compare_exchange_impl(
        atomic_ref, expected_ref, desired, expected_idx=0
    ):
        # pylint: disable=no-value-for-parameter
        return _intrinsic_compare_exchange(
            atomic_ref, expected_ref, desired, expected_idx
        )

    return ol_compare_exchange_impl
