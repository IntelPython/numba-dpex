# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir as llvmir
from numba import errors
from numba.core import cgutils, types
from numba.extending import intrinsic, overload, overload_method

from numba_dpex.core import itanium_mangler as ext_itanium_mangler
from numba_dpex.core.targets.kernel_target import (
    CC_SPIR_FUNC,
    DPEX_KERNEL_TARGET_NAME,
)
from numba_dpex.core.types import USMNdArray

from ..dpcpp_types import AtomicRefType
from ._spv_atomic_helper import (
    get_atomic_inst_name,
    get_memory_semantics_mask,
    get_scope,
)


def _parse_int_literal(literal_int: types.scalars.IntegerLiteral) -> int:
    """Parse an instance of a numba.core.types.Literal to its actual int value.

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
    else:
        raise errors.TypingError(
            "The parameter 'literal_int' is not an instance of "
            + "types.IntegerLiteral"
        )


class AtomicRef:
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

    def fetch_sub(self, val):
        pass

    def fetch_min(self, val):
        pass

    def fetch_max(self, val):
        pass

    def fetch_and(self, val):
        pass

    def fetch_or(self, val):
        pass

    def fetch_xor(self, val):
        pass

    def load(self):
        pass

    def store(self, val):
        pass

    def exchange(self, val):
        pass

    def compare_exchange_weak(self, expected, desired):
        pass

    def compare_exchange_strong(self, expected, desired):
        pass


def _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, op_str):
    from ..target import dpex_exp_kernel_target

    sig = types.void(ty_atomic_ref, ty_val)

    def gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_ref_dtype = atomic_ref_ty.dtype

        ref = args[0]
        dmm = dpex_exp_kernel_target.target_context.data_model_manager
        data_attr_pos = dmm.lookup(sig.args[0]).get_field_position("ref")

        data_attr = builder.extract_value(ref, data_attr_pos)

        # if atomic_ref_dtype in (types.float32, types.float64):
        # XXX Turn on once the overload is added to DpexKernelTarget
        # context.extra_compile_options[context.LLVM_SPIRV_ARGS] = [
        #     "--spirv-ext=+SPV_EXT_shader_atomic_float_add"
        # ]

        name = get_atomic_inst_name(op_str, atomic_ref_dtype)

        ptr_type = context.get_value_type(atomic_ref_dtype).as_pointer()
        ptr_type.addrspace = atomic_ref_ty.address_space
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

        fn.calling_convention = CC_SPIR_FUNC
        spirv_memory_semantics_mask = get_memory_semantics_mask(
            atomic_ref_ty.memory_order
        )
        spirv_scope = get_scope(atomic_ref_ty.memory_scope)

        fn_args = [
            data_attr,
            context.get_constant(types.int32, spirv_scope),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
            args[1],
        ]

        builder.call(fn, fn_args)

    return sig, gen


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_fetch_add(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_add")


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_fetch_sub(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_sub")


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_fetch_min(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_min")


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_fetch_max(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_max")


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_fetch_and(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_and")


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_fetch_or(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_or")


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_fetch_xor(ty_context, ty_atomic_ref, ty_val):
    return _intrinsic_helper(ty_context, ty_atomic_ref, ty_val, "fetch_xor")


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_load(ty_context, ty_atomic_ref):
    from ..target import dpex_exp_kernel_target

    sig = types.void(ty_atomic_ref)

    def gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_ref_dtype = atomic_ref_ty.dtype

        ref = args[0]
        dmm = dpex_exp_kernel_target.target_context.data_model_manager
        data_attr_pos = dmm.lookup(sig.args[0]).get_field_position("ref")

        data_attr = builder.extract_value(ref, data_attr_pos)

        # if atomic_ref_dtype in (types.float32, types.float64):
        # XXX Turn on once the overload is added to DpexKernelTarget
        # context.extra_compile_options[context.LLVM_SPIRV_ARGS] = [
        #     "--spirv-ext=+SPV_EXT_shader_atomic_float_add"
        # ]

        name = "__spirv_AtomicLoad"

        ptr_type = context.get_value_type(atomic_ref_dtype).as_pointer()
        ptr_type.addrspace = atomic_ref_ty.address_space.literal_value
        retty = context.get_value_type(atomic_ref_dtype)
        spirv_fn_arg_types = [
            ptr_type,
            llvmir.IntType(32),
            llvmir.IntType(32),
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
            ],
        )
        fnty = llvmir.FunctionType(retty, spirv_fn_arg_types)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, mangled_fn_name
        )
        spirv_memory_semantics_mask = get_memory_semantics_mask(
            atomic_ref_ty.memory_order.literal_value
        )
        spirv_scope = get_scope(atomic_ref_ty.memory_scope.literal_value)

        fn_args = [
            data_attr,
            context.get_constant(types.int32, spirv_scope),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
        ]

        builder.call(fn, fn_args)

    return sig, gen


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_store(ty_context, ty_atomic_ref, ty_val):
    from ..target import dpex_exp_kernel_target

    sig = types.void(ty_atomic_ref, ty_val)

    def gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_ref_dtype = atomic_ref_ty.dtype

        ref = args[0]
        dmm = dpex_exp_kernel_target.target_context.data_model_manager
        data_attr_pos = dmm.lookup(sig.args[0]).get_field_position("ref")

        data_attr = builder.extract_value(ref, data_attr_pos)

        # if atomic_ref_dtype in (types.float32, types.float64):
        # XXX Turn on once the overload is added to DpexKernelTarget
        # context.extra_compile_options[context.LLVM_SPIRV_ARGS] = [
        #     "--spirv-ext=+SPV_EXT_shader_atomic_float_add"
        # ]

        name = "__spirv_AtomicStore"

        ptr_type = context.get_value_type(atomic_ref_dtype).as_pointer()
        ptr_type.addrspace = atomic_ref_ty.address_space.literal_value
        retty = llvmir.VoidType()
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
        fn.calling_convention = CC_SPIR_FUNC
        spirv_memory_semantics_mask = get_memory_semantics_mask(
            atomic_ref_ty.memory_order.literal_value
        )
        spirv_scope = get_scope(atomic_ref_ty.memory_scope.literal_value)

        fn_args = [
            data_attr,
            context.get_constant(types.int32, spirv_scope),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
            args[1],
        ]

        builder.call(fn, fn_args)

    return sig, gen


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_exchange(ty_context, ty_atomic_ref, ty_val):
    from ..target import dpex_exp_kernel_target

    sig = types.void(ty_atomic_ref, ty_val)

    def gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_ref_dtype = atomic_ref_ty.dtype

        ref = args[0]
        dmm = dpex_exp_kernel_target.target_context.data_model_manager
        data_attr_pos = dmm.lookup(sig.args[0]).get_field_position("ref")

        data_attr = builder.extract_value(ref, data_attr_pos)

        # if atomic_ref_dtype in (types.float32, types.float64):
        # XXX Turn on once the overload is added to DpexKernelTarget
        # context.extra_compile_options[context.LLVM_SPIRV_ARGS] = [
        #     "--spirv-ext=+SPV_EXT_shader_atomic_float_add"
        # ]

        name = "__spirv_AtomicExchange"

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

        fn.calling_convention = CC_SPIR_FUNC
        spirv_memory_semantics_mask = get_memory_semantics_mask(
            atomic_ref_ty.memory_order.literal_value
        )
        spirv_scope = get_scope(atomic_ref_ty.memory_scope.literal_value)

        fn_args = [
            data_attr,
            context.get_constant(types.int32, spirv_scope),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
            args[1],
        ]

        builder.call(fn, fn_args)

    return sig, gen


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_compare_exchange(
    ty_context, ty_atomic_ref, ty_expected, ty_desired
):
    from ..target import dpex_exp_kernel_target

    sig = types.void(ty_atomic_ref, ty_expected, ty_desired)

    def gen(context, builder, sig, args):
        atomic_ref_ty = sig.args[0]
        atomic_ref_dtype = atomic_ref_ty.dtype

        ref = args[0]
        dmm = dpex_exp_kernel_target.target_context.data_model_manager
        data_attr_pos = dmm.lookup(sig.args[0]).get_field_position("ref")

        data_attr = builder.extract_value(ref, data_attr_pos)

        # if atomic_ref_dtype in (types.float32, types.float64):
        # XXX Turn on once the overload is added to DpexKernelTarget
        # context.extra_compile_options[context.LLVM_SPIRV_ARGS] = [
        #     "--spirv-ext=+SPV_EXT_shader_atomic_float_add"
        # ]

        name = "__spirv_AtomicCompareExchange"

        ptr_type = context.get_value_type(atomic_ref_dtype).as_pointer()
        ptr_type.addrspace = atomic_ref_ty.address_space.literal_value
        retty = context.get_value_type(atomic_ref_dtype)
        spirv_fn_arg_types = [
            ptr_type,
            llvmir.IntType(32),
            llvmir.IntType(32),
            llvmir.IntType(32),
            context.get_value_type(atomic_ref_dtype),
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
                atomic_ref_dtype,
            ],
        )
        fnty = llvmir.FunctionType(retty, spirv_fn_arg_types)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, mangled_fn_name
        )
        fn.calling_convention = CC_SPIR_FUNC
        spirv_memory_semantics_mask = get_memory_semantics_mask(
            atomic_ref_ty.memory_order.literal_value
        )
        spirv_scope = get_scope(atomic_ref_ty.memory_scope.literal_value)

        fn_args = [
            data_attr,
            context.get_constant(types.int32, spirv_scope),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
            context.get_constant(types.int32, spirv_memory_semantics_mask),
            args[1],
            args[2],
        ]

        builder.call(fn, fn_args)

        return

    return sig, gen


@intrinsic(target=DPEX_KERNEL_TARGET_NAME)
def _intrinsic_atomic_ref_ctor(ty_context, ref, ty_retty_ref):
    from ..target import dpex_exp_kernel_target

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(ref, ty_retty_ref)

    def codegen(context, builder, sig, args):
        ref = args[0]
        dmm = dpex_exp_kernel_target.target_context.data_model_manager
        data_attr_pos = dmm.lookup(sig.args[0]).get_field_position("data")

        data_attr = builder.extract_value(ref, data_attr_pos)
        atomic_ref_struct = cgutils.create_struct_proxy(ty_retty)(
            context, builder
        )

        # Populate the atomic ref data model
        ref_attr_pos = dmm.lookup(ty_retty).get_field_position("ref")
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

    if not isinstance(ref, USMNdArray):
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
    elif ref.dtype in [types.int64, types.float64, types.uint64]:
        if not ref.queue.device_has_aspect_atomic64:
            raise errors.TypingError(
                "Targeted device does not support 64-bit atomic operations."
            )
    else:
        return supported


@overload(
    AtomicRef,
    prefer_literal=True,
    inline="always",
    target=DPEX_KERNEL_TARGET_NAME,
)
def ol_atomic_ref(ref, memory_order, memory_scope, address_space):
    _check_if_supported_ref(ref)

    _address_space = _parse_int_literal(address_space)
    _memory_order = _parse_int_literal(memory_order)
    _memory_scope = _parse_int_literal(memory_scope)

    ty_retty = AtomicRefType(
        dtype=ref.dtype,
        memory_order=_memory_order,
        memory_scope=_memory_scope,
        address_space=_address_space,
        has_aspect_atomic64=ref.queue.device_has_aspect_atomic64,
    )

    def ol_atomic_ref_ctor_impl(ref, memory_order, memory_scope, address_space):
        return _intrinsic_atomic_ref_ctor(ref, ty_retty)

    return ol_atomic_ref_ctor_impl


@overload_method(
    AtomicRefType, "fetch_add", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_fetch_add(atomic_ref, val):
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to add: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_fetch_add_impl(atomic_ref, val):
        return _intrinsic_fetch_add(atomic_ref, val)

    return ol_fetch_add_impl


@overload_method(
    AtomicRefType, "fetch_sub", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_fetch_sub(atomic_ref, val):
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to sub: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_fetch_sub_impl(atomic_ref, val):
        return _intrinsic_fetch_sub(atomic_ref, val)

    return ol_fetch_sub_impl


@overload_method(
    AtomicRefType, "fetch_min", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_fetch_min(atomic_ref, val):
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to find min: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_fetch_min_impl(atomic_ref, val):
        return _intrinsic_fetch_min(atomic_ref, val)

    return ol_fetch_min_impl


@overload_method(
    AtomicRefType, "fetch_max", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_fetch_max(atomic_ref, val):
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to find max: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_fetch_max_impl(atomic_ref, val):
        return _intrinsic_fetch_max(atomic_ref, val)

    return ol_fetch_max_impl


@overload_method(
    AtomicRefType, "fetch_and", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_fetch_and(atomic_ref, val):
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
        return _intrinsic_fetch_and(atomic_ref, val)

    return ol_fetch_and_impl


@overload_method(
    AtomicRefType, "fetch_or", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_fetch_or(atomic_ref, val):
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
        return _intrinsic_fetch_or(atomic_ref, val)

    return ol_fetch_or_impl


@overload_method(
    AtomicRefType, "fetch_xor", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_fetch_xor(atomic_ref, val):
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
        return _intrinsic_fetch_xor(atomic_ref, val)

    return ol_fetch_xor_impl


@overload_method(
    AtomicRefType, "load", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_load(atomic_ref):
    def ol_load_impl(atomic_ref):
        return _intrinsic_load(atomic_ref)

    return ol_load_impl


@overload_method(
    AtomicRefType, "store", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_store(atomic_ref, val):
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to store: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_store_impl(atomic_ref, val):
        return _intrinsic_store(atomic_ref, val)

    return ol_store_impl


@overload_method(
    AtomicRefType, "exchange", inline="always", target=DPEX_KERNEL_TARGET_NAME
)
def ol_exchange(atomic_ref, val):
    if atomic_ref.dtype != val:
        raise errors.TypingError(
            f"Type of value to exchange: {val} does not match the type of the "
            f"reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_exchange_impl(atomic_ref, val):
        return _intrinsic_exchange(atomic_ref, val)

    return ol_exchange_impl


@overload_method(
    AtomicRefType,
    "compare_exchange_weak",
    inline="always",
    target=DPEX_KERNEL_TARGET_NAME,
)
def ol_compare_exchange_weak(atomic_ref, expected_ref, desired):
    if atomic_ref.dtype != expected_ref:
        raise errors.TypingError(
            f"Type of value to compare_exchange_weak: {expected_ref} does not match the "
            f"type of the reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    if atomic_ref.dtype != desired:
        raise errors.TypingError(
            f"Type of value to compare_exchange_strong: {desired} does not match the "
            f"type of the reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_compare_exchange_weak_impl(atomic_ref, expected_ref, desired):
        return _intrinsic_compare_exchange(atomic_ref, expected_ref, desired)

    return ol_compare_exchange_weak_impl


@overload_method(
    AtomicRefType,
    "compare_exchange_strong",
    inline="always",
    target=DPEX_KERNEL_TARGET_NAME,
)
def ol_compare_exchange_strong(atomic_ref, expected_ref, desired):
    if atomic_ref.dtype != expected_ref:
        raise errors.TypingError(
            f"Type of value to compare_exchange_strong: {expected_ref} does not match the "
            f"type of the reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    if atomic_ref.dtype != desired:
        raise errors.TypingError(
            f"Type of value to compare_exchange_strong: {desired} does not match the "
            f"type of the reference: {atomic_ref.dtype} stored in the atomic ref."
        )

    def ol_compare_exchange_strong_impl(atomic_ref, expected_ref, desired):
        return _intrinsic_compare_exchange(atomic_ref, expected_ref, desired)

    return ol_compare_exchange_strong_impl
