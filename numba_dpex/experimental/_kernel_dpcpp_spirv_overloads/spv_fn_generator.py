# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements a set of helper functions to generate the LLVM IR for SPIR-V
functions and their use inside an LLVM module.
"""

from llvmlite import ir as llvmir
from numba.core import cgutils, types

from numba_dpex.core import itanium_mangler as ext_itanium_mangler
from numba_dpex.kernel_api_impl.spirv.target import CC_SPIR_FUNC


def get_or_insert_atomic_load_fn(context, module, atomic_ref_ty):
    """
    Gets or inserts a declaration for a __spirv_AtomicLoad call into the
    specified LLVM IR module.
    """
    atomic_ref_dtype = atomic_ref_ty.dtype
    atomic_load_fn_retty = context.get_value_type(atomic_ref_dtype)
    ptr_type = atomic_load_fn_retty.as_pointer()
    ptr_type.addrspace = atomic_ref_ty.address_space
    atomic_load_fn_arg_types = [
        ptr_type,
        llvmir.IntType(32),
        llvmir.IntType(32),
    ]
    mangled_fn_name = ext_itanium_mangler.mangle_ext(
        "__spirv_AtomicLoad",
        [
            types.CPointer(atomic_ref_dtype, addrspace=ptr_type.addrspace),
            "__spv.Scope.Flag",
            "__spv.MemorySemanticsMask.Flag",
        ],
    )

    fn = cgutils.get_or_insert_function(
        module,
        llvmir.FunctionType(atomic_load_fn_retty, atomic_load_fn_arg_types),
        mangled_fn_name,
    )
    fn.calling_convention = CC_SPIR_FUNC

    return fn


def get_or_insert_spv_atomic_store_fn(context, module, atomic_ref_ty):
    """
    Gets or inserts a declaration for a __spirv_AtomicStore call into the
    specified LLVM IR module.
    """
    atomic_ref_dtype = atomic_ref_ty.dtype
    ptr_type = context.get_value_type(atomic_ref_dtype).as_pointer()
    ptr_type.addrspace = atomic_ref_ty.address_space
    atomic_store_fn_retty = llvmir.VoidType()
    atomic_store_fn_arg_types = [
        ptr_type,
        llvmir.IntType(32),
        llvmir.IntType(32),
        context.get_value_type(atomic_ref_dtype),
    ]

    mangled_fn_name = ext_itanium_mangler.mangle_ext(
        "__spirv_AtomicStore",
        [
            types.CPointer(atomic_ref_dtype, addrspace=ptr_type.addrspace),
            "__spv.Scope.Flag",
            "__spv.MemorySemanticsMask.Flag",
            atomic_ref_dtype,
        ],
    )

    fn = cgutils.get_or_insert_function(
        module,
        llvmir.FunctionType(atomic_store_fn_retty, atomic_store_fn_arg_types),
        mangled_fn_name,
    )
    fn.calling_convention = CC_SPIR_FUNC

    return fn


def get_or_insert_spv_atomic_exchange_fn(context, module, atomic_ref_ty):
    """
    Gets or inserts a declaration for a __spirv_AtomicExchange call into the
    specified LLVM IR module.
    """
    atomic_ref_dtype = atomic_ref_ty.dtype
    ptr_type = context.get_value_type(atomic_ref_dtype).as_pointer()
    ptr_type.addrspace = atomic_ref_ty.address_space
    atomic_exchange_fn_retty = context.get_value_type(atomic_ref_ty.dtype)
    atomic_exchange_fn_arg_types = [
        ptr_type,
        llvmir.IntType(32),
        llvmir.IntType(32),
        context.get_value_type(atomic_ref_dtype),
    ]

    mangled_fn_name = ext_itanium_mangler.mangle_ext(
        "__spirv_AtomicExchange",
        [
            types.CPointer(atomic_ref_dtype, addrspace=ptr_type.addrspace),
            "__spv.Scope.Flag",
            "__spv.MemorySemanticsMask.Flag",
            atomic_ref_dtype,
        ],
    )

    fn = cgutils.get_or_insert_function(
        module,
        llvmir.FunctionType(
            atomic_exchange_fn_retty, atomic_exchange_fn_arg_types
        ),
        mangled_fn_name,
    )
    fn.calling_convention = CC_SPIR_FUNC

    return fn


def get_or_insert_spv_atomic_compare_exchange_fn(
    context, module, atomic_ref_ty
):
    """
    Gets or inserts a declaration for a __spirv_AtomicCompareExchange call into the
    specified LLVM IR module.
    """
    atomic_ref_dtype = atomic_ref_ty.dtype

    # Spirv spec requires arguments and return type to be of integer types.
    # That is why the type is changed from float to int
    # while maintaining the bit-width.
    # During function call, bitcasting is performed
    # to adhere to this convention.
    if atomic_ref_dtype == types.float32:
        atomic_ref_dtype = types.uint32
    elif atomic_ref_dtype == types.float64:
        atomic_ref_dtype = types.uint64

    ptr_type = context.get_value_type(atomic_ref_dtype).as_pointer()
    ptr_type.addrspace = atomic_ref_ty.address_space
    atomic_cmpexchg_fn_retty = context.get_value_type(atomic_ref_dtype)

    atomic_cmpexchg_fn_arg_types = [
        ptr_type,
        llvmir.IntType(32),
        llvmir.IntType(32),
        llvmir.IntType(32),
        context.get_value_type(atomic_ref_dtype),
        context.get_value_type(atomic_ref_dtype),
    ]

    mangled_fn_name = ext_itanium_mangler.mangle_ext(
        "__spirv_AtomicCompareExchange",
        [
            types.CPointer(atomic_ref_dtype, addrspace=ptr_type.addrspace),
            "__spv.Scope.Flag",
            "__spv.MemorySemanticsMask.Flag",
            "__spv.MemorySemanticsMask.Flag",
            atomic_ref_dtype,
            atomic_ref_dtype,
        ],
    )

    fn = cgutils.get_or_insert_function(
        module,
        llvmir.FunctionType(
            atomic_cmpexchg_fn_retty, atomic_cmpexchg_fn_arg_types
        ),
        mangled_fn_name,
    )
    fn.calling_convention = CC_SPIR_FUNC

    return fn
