# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
An implementation of ``print`` for use in a kernel for the SPIRVKernelTarget.
"""

from functools import singledispatch

import llvmlite.ir as llvmir
from numba.core import cgutils, types
from numba.core.imputils import Registry

from numba_dpex.kernel_api.memory_enums import AddressSpace as address_space

registry = Registry()
lower = registry.lower


def declare_print(lmod: llvmir.Module):
    """Inserts declaration for C printf into the given LLVM module

    Args:
        lmod (llvmir.Module): LLVM module into which the function declaration
            needs to be inserted.

    Returns:
        An LLVM IR Function object for the inserted C printf function.
    """
    voidptrty = llvmir.PointerType(
        llvmir.IntType(8), addrspace=address_space.GENERIC.value
    )
    printfty = llvmir.FunctionType(
        llvmir.IntType(32), [voidptrty], var_arg=True
    )
    printf = cgutils.get_or_insert_function(lmod, printfty, "printf")
    return printf


@singledispatch
def print_item(ty, context, builder, val):
    """
    Handle printing of a single value of the given Numba type.
    A (format string, [list of arguments]) is returned that will allow
    forming the final printf()-like call.
    """
    raise NotImplementedError(f"printing unimplemented for values of type {ty}")


@print_item.register(types.Integer)
@print_item.register(types.IntegerLiteral)
def int_print_impl(ty, context, builder, val):
    """Implements printing an integer value."""
    if ty in types.unsigned_domain:
        rawfmt = "%llu"
        dsttype = types.uint64
    else:
        rawfmt = "%lld"
        dsttype = types.int64
    context.insert_const_string(builder.module, rawfmt)
    lld = context.cast(builder, val, ty, dsttype)
    return rawfmt, [lld]


@print_item.register(types.Float)
def real_print_impl(ty, context, builder, val):
    """Implements printing a real number value."""
    lld = context.cast(builder, val, ty, types.float64)
    return "%f", [lld]


@print_item.register(types.StringLiteral)
def const_print_impl(ty, context, builder, sigval):
    """Implements printing a string value."""
    pyval = ty.literal_value
    assert isinstance(pyval, str)  # Ensured by lowering
    rawfmt = "%s"
    val = context.insert_const_string(builder.module, pyval)
    return rawfmt, [val]


@lower(print, types.VarArg(types.Any))
def print_varargs(context, builder, sig, args):
    """This function is a generic 'print' wrapper for arbitrary types.
    It dispatches to the appropriate 'print' implementations above
    depending on the detected real types in the signature."""

    formats = []
    values = []

    only_str = True
    for _, (argtype, argval) in enumerate(zip(sig.args, args)):
        argfmt, argvals = print_item(argtype, context, builder, argval)
        formats.append(argfmt)
        values.extend(argvals)
        if argfmt != "%s":
            only_str = False

    if only_str:
        raise ValueError("We do not support printing string alone!")

    rawfmt = " ".join(formats) + "\n"
    fmt = context.insert_const_string(builder.module, rawfmt)

    va_arg = [fmt]
    va_arg.extend(values)
    va_arg = tuple(va_arg)

    print_fn = declare_print(builder.module)

    builder.call(print_fn, va_arg)

    return context.get_dummy_value()
