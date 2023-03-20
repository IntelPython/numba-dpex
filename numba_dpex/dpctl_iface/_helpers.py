# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import types


def numba_type_to_dpctl_typenum(context, type):
    """
    This function looks up the dpctl defined enum values from
    ``DPCTLKernelArgType``.
    """

    val = None
    if type == types.int32 or isinstance(type, types.scalars.IntegerLiteral):
        # DPCTL_LONG_LONG
        val = context.get_constant(types.int32, 9)
    elif type == types.uint32:
        # DPCTL_UNSIGNED_LONG_LONG
        val = context.get_constant(types.int32, 10)
    elif type == types.boolean:
        # DPCTL_UNSIGNED_INT
        val = context.get_constant(types.int32, 5)
    elif type == types.int64:
        # DPCTL_LONG_LONG
        val = context.get_constant(types.int32, 9)
    elif type == types.uint64:
        # DPCTL_SIZE_T
        val = context.get_constant(types.int32, 11)
    elif type == types.float32:
        # DPCTL_FLOAT
        val = context.get_constant(types.int32, 12)
    elif type == types.float64:
        # DPCTL_DOUBLE
        val = context.get_constant(types.int32, 13)
    elif type == types.voidptr:
        # DPCTL_VOID_PTR
        val = context.get_constant(types.int32, 15)
    else:
        raise NotImplementedError

    return val
