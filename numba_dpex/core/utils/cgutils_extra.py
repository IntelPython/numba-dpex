# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir as llvmir
from numba.core import cgutils, types

from numba_dpex.core.utils.itanium_mangler import mangle_c


class LLVMTypes:
    """
    A helper class to get LLVM Values for integer C types.
    """

    byte_t = llvmir.IntType(8)
    byte_ptr_t = byte_t.as_pointer()
    byte_ptr_ptr_t = byte_ptr_t.as_pointer()
    int32_t = llvmir.IntType(32)
    int32_ptr_t = int32_t.as_pointer()
    int64_t = llvmir.IntType(64)
    int64_ptr_t = int64_t.as_pointer()
    void_t = llvmir.VoidType()


def declare_function(context, builder, name, sig, cargs, mangler=mangle_c):
    """Insert declaration for a opencl builtin function.
    Uses the Itanium mangler.

    Args
    ----
    context: target context

    builder: llvm builder

    name: str
        symbol name

    sig: signature
        function signature of the symbol being declared

    cargs: sequence of str
        C type names for the arguments

    mangler: a mangler function
        function to use to mangle the symbol

    """
    mod = builder.module
    if sig.return_type == types.void:
        llretty = llvmir.VoidType()
    else:
        llretty = context.get_value_type(sig.return_type)
    llargs = [context.get_value_type(t) for t in sig.args]
    fnty = llvmir.FunctionType(llretty, llargs)
    mangled = mangler(name, cargs)
    fn = cgutils.get_or_insert_function(mod, fnty, mangled)
    from numba_dpex import spirv_kernel_target

    fn.calling_convention = spirv_kernel_target.CC_SPIR_FUNC
    return fn


def get_llvm_type(context, type):
    """Returns the LLVM Value corresponding to a Numba type.

    Args:
        context: The LLVM context or the execution state of the current IR
        generator.
        type: A Numba type object.

    Returns: An Python object wrapping an LLVM Value corresponding to the
             specified Numba type.

    """
    return context.get_value_type(type)


def get_llvm_ptr_type(type):
    """Returns an LLVM pointer type for a give LLVM type object.

    Args:
        type: An LLVM type for which we need the corresponding pointer type.

    Returns: An LLVM pointer type object corresponding to the input LLVM type.

    """
    return type.as_pointer()


def create_null_ptr(builder, context):
    """
    Allocates a new LLVM Value storing a ``void*`` and returns the Value to
    caller.

    Args:
        builder: The LLVM IR builder to be used for code generation.
        context: The LLVM IR builder context.

    Returns: An LLVM value storing a null pointer

    """
    null_ptr = cgutils.alloca_once(
        builder=builder,
        ty=context.get_value_type(types.voidptr),
        size=context.get_constant(types.uintp, 1),
    )
    builder.store(
        builder.inttoptr(
            context.get_constant(types.uintp, 0),
            get_llvm_type(context=context, type=types.voidptr),
        ),
        null_ptr,
    )
    return null_ptr


def get_zero(context):
    """Returns an LLVM Constant storing a 64 bit representation for zero.

    Args:
        context: The LLVM IR builder context.

    Returns: An LLVM Constant Value storing zero.

    """
    return context.get_constant(types.uintp, 0)


def get_one(context):
    """Returns an LLVM Constant storing a 64 bit representation for one.

    Args:
        context: The LLVM IR builder context.

    Returns: An LLVM Constant Value storing one.

    """
    return context.get_constant(types.uintp, 1)
