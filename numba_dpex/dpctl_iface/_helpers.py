# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import types

from numba_dpex import dpctl_sem_version


def numba_type_to_dpctl_typenum(context, ty):
    """
    This function looks up the dpctl defined enum values from
    ``DPCTLKernelArgType``.
    """

    if dpctl_sem_version >= (0, 17, 0):
        # FIXME change to imports from a dpctl enum/class rather than
        # hard coding these numbers.

        if ty == types.boolean:
            return context.get_constant(types.int32, 1)
        elif ty == types.int32 or isinstance(ty, types.scalars.IntegerLiteral):
            return context.get_constant(types.int32, 4)
        elif ty == types.uint32:
            return context.get_constant(types.int32, 5)
        elif ty == types.int64:
            return context.get_constant(types.int32, 6)
        elif ty == types.uint64:
            return context.get_constant(types.int32, 7)
        elif ty == types.float32:
            return context.get_constant(types.int32, 8)
        elif ty == types.float64:
            return context.get_constant(types.int32, 9)
        elif ty == types.voidptr or isinstance(ty, types.CPointer):
            return context.get_constant(types.int32, 10)
        else:
            raise NotImplementedError
    else:
        if ty == types.int32 or isinstance(ty, types.scalars.IntegerLiteral):
            # DPCTL_LONG_LONG
            return context.get_constant(types.int32, 9)
        elif ty == types.uint32:
            # DPCTL_UNSIGNED_LONG_LONG
            return context.get_constant(types.int32, 10)
        elif ty == types.boolean:
            # DPCTL_UNSIGNED_INT
            return context.get_constant(types.int32, 5)
        elif ty == types.int64:
            # DPCTL_LONG_LONG
            return context.get_constant(types.int32, 9)
        elif ty == types.uint64:
            # DPCTL_SIZE_T
            return context.get_constant(types.int32, 11)
        elif ty == types.float32:
            # DPCTL_FLOAT
            return context.get_constant(types.int32, 12)
        elif ty == types.float64:
            # DPCTL_DOUBLE
            return context.get_constant(types.int32, 13)
        elif ty == types.voidptr or isinstance(ty, types.CPointer):
            # DPCTL_VOID_PTR
            return context.get_constant(types.int32, 15)
        else:
            raise NotImplementedError
