# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import types

from numba_dpex import dpctl_sem_version
from numba_dpex.core.types.kernel_api.local_accessor import LocalAccessorType


def numba_type_to_dpctl_typenum(context, ty):
    """
    This function looks up the dpctl defined enum values from
    ``DPCTLKernelArgType``.
    """

    if dpctl_sem_version >= (0, 17, 0):
        from dpctl._sycl_queue import kernel_arg_type as kargty

        if ty == types.boolean:
            return context.get_constant(types.int32, kargty.dpctl_uint8.value)
        elif ty == types.int32 or isinstance(ty, types.scalars.IntegerLiteral):
            return context.get_constant(types.int32, kargty.dpctl_int32.value)
        elif ty == types.uint32:
            return context.get_constant(types.int32, kargty.dpctl_uint32.value)
        elif ty == types.int64:
            return context.get_constant(types.int32, kargty.dpctl_int64.value)
        elif ty == types.uint64:
            return context.get_constant(types.int32, kargty.dpctl_uint64.value)
        elif ty == types.float32:
            return context.get_constant(types.int32, kargty.dpctl_float32.value)
        elif ty == types.float64:
            return context.get_constant(types.int32, kargty.dpctl_float64.value)
        elif ty == types.voidptr or isinstance(ty, types.CPointer):
            return context.get_constant(
                types.int32, kargty.dpctl_void_ptr.value
            )
        elif isinstance(ty, LocalAccessorType):
            return context.get_constant(
                types.int32, kargty.dpctl_local_accessor.value
            )
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
        elif isinstance(ty, LocalAccessorType):
            raise NotImplementedError(
                "LocalAccessor args for kernels requires dpctl 0.17 or greater."
            )
        else:
            raise NotImplementedError
