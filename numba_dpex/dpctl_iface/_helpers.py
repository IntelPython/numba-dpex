# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import types

from numba_dpex.core.types.kernel_api.local_accessor import LocalAccessorType


def numba_type_to_dpctl_typenum(context, ty):
    """
    This function looks up the dpctl defined enum values from
    ``DPCTLKernelArgType``.
    """

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
        return context.get_constant(types.int32, kargty.dpctl_void_ptr.value)
    elif isinstance(ty, LocalAccessorType):
        return context.get_constant(
            types.int32, kargty.dpctl_local_accessor.value
        )
    else:
        raise NotImplementedError
