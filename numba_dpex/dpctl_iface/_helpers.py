# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import types


def numba_type_to_dpctl_typenum(context, ty):
    """
    This function looks up the dpctl defined enum values from
    ``DPCTLKernelArgType``.
    """

    val = None
    if ty == types.int32 or isinstance(ty, types.scalars.IntegerLiteral):
        # DPCTL_LONG_LONG
        val = context.get_constant(types.int32, 9)
    elif ty == types.uint32:
        # DPCTL_UNSIGNED_LONG_LONG
        val = context.get_constant(types.int32, 10)
    elif ty == types.boolean:
        # DPCTL_UNSIGNED_INT
        val = context.get_constant(types.int32, 5)
    elif ty == types.int64:
        # DPCTL_LONG_LONG
        val = context.get_constant(types.int32, 9)
    elif ty == types.uint64:
        # DPCTL_SIZE_T
        val = context.get_constant(types.int32, 11)
    elif ty == types.float32:
        # DPCTL_FLOAT
        val = context.get_constant(types.int32, 12)
    elif ty == types.float64:
        # DPCTL_DOUBLE
        val = context.get_constant(types.int32, 13)
    elif ty == types.voidptr or isinstance(ty, types.CPointer):
        # DPCTL_VOID_PTR
        val = context.get_constant(types.int32, 15)
    else:
        raise NotImplementedError

    return val
