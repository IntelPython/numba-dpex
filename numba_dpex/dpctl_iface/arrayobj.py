# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import operator

from llvmlite.ir import IRBuilder
from numba.core import cgutils, errors, imputils, types
from numba.core.imputils import impl_ret_borrowed
from numba.extending import intrinsic, overload_attribute
from numba.np.arrayobj import _getitem_array_generic as np_getitem_array_generic
from numba.np.arrayobj import make_array

from numba_dpex.core.types import DpnpNdArray, USMNdArray
from numba_dpex.core.types.dpctl_types import DpctlSyclQueue
from numba_dpex.kernel_api_impl.spirv.arrayobj import (
    _getitem_array_generic as kernel_getitem_array_generic,
)
from numba_dpex.kernel_api_impl.spirv.target import SPIRVTargetContext

from .dpctlimpl import lower_builtin

# can't import name because of the circular import
DPEX_TARGET_NAME = "dpex"

# =========================================================================
#               Helps to parse dpnp constructor arguments
# =========================================================================


# TODO: target specific
@lower_builtin(operator.getitem, USMNdArray, types.Integer)
@lower_builtin(operator.getitem, USMNdArray, types.SliceType)
def getitem_arraynd_intp(context, builder, sig, args):
    """
    Overrding the numba.np.arrayobj.getitem_arraynd_intp to support dpnp.ndarray

    The data model for numba.types.Array and numba_dpex.types.DpnpNdArray
    are different. DpnpNdArray has an extra attribute to store a sycl::queue
    pointer. For that reason, np_getitem_arraynd_intp needs to be overriden so
    that when returning a view of a dpnp.ndarray the sycl::queue pointer
    member in the LLVM IR struct gets properly updated.
    """
    getitem_call_in_kernel = isinstance(context, SPIRVTargetContext)
    _getitem_array_generic = np_getitem_array_generic

    if getitem_call_in_kernel:
        _getitem_array_generic = kernel_getitem_array_generic

    aryty, idxty = sig.args
    ary, idx = args

    assert aryty.ndim >= 1
    ary = make_array(aryty)(context, builder, ary)

    res = _getitem_array_generic(
        context, builder, sig.return_type, aryty, ary, (idxty,), (idx,)
    )
    ret = impl_ret_borrowed(context, builder, sig.return_type, res)

    if isinstance(sig.return_type, USMNdArray) and not getitem_call_in_kernel:
        array_val = args[0]
        array_ty = sig.args[0]
        sycl_queue_attr_pos = context.data_model_manager.lookup(
            array_ty
        ).get_field_position("sycl_queue")
        sycl_queue_attr = builder.extract_value(array_val, sycl_queue_attr_pos)
        ret = builder.insert_value(ret, sycl_queue_attr, sycl_queue_attr_pos)

    return ret


@intrinsic(target=DPEX_TARGET_NAME)
def ol_usm_nd_array_sycl_queue(
    ty_context,
    ty_dpnp_nd_array: DpnpNdArray,
):
    if not isinstance(ty_dpnp_nd_array, DpnpNdArray):
        raise errors.TypingError("Argument must be DpnpNdArray")

    ty_queue: DpctlSyclQueue = ty_dpnp_nd_array.queue

    sig = ty_queue(ty_dpnp_nd_array)

    def codegen(context, builder: IRBuilder, sig, args: list):
        array_proxy = cgutils.create_struct_proxy(ty_dpnp_nd_array)(
            context,
            builder,
            value=args[0],
        )

        queue_ref = array_proxy.sycl_queue

        queue_struct_proxy = cgutils.create_struct_proxy(ty_queue)(
            context, builder
        )

        queue_struct_proxy.queue_ref = queue_ref
        queue_struct_proxy.meminfo = array_proxy.meminfo

        # Warning: current implementation prevents whole object from being
        # destroyed as long as sycl_queue attribute is being used. It should be
        # okay since anywere we use it as an argument callee creates a copy
        # so it does not steel reference.
        #
        # We can avoid it by:
        #  queue_ref_copy = sycl.dpctl_queue_copy(builder, queue_ref) #noqa E800
        #  queue_struct_proxy.queue_ref = queue_ref_copy #noqa E800
        #  queue_struct->meminfo =
        #     nrt->manage_memory(queue_ref_copy, DPCTLEvent_Delete);
        # but it will allocate new meminfo object which can negatively affect
        # performance.
        # Speaking philosophically attribute is a part of the object and as long
        # as nobody can still the reference it is a part of the owner object
        # and lifetime is tied to it.
        # TODO: we want to have queue: queuestruct_t instead of
        #   queue_ref: QueueRef as an attribute for DPNPNdArray.

        queue_value = queue_struct_proxy._getvalue()

        # We need to incref meminfo so that queue model is preventing parent
        # ndarray from being destroyed, that can destroy queue that we are
        # using.
        return imputils.impl_ret_borrowed(
            context, builder, ty_queue, queue_value
        )

    return sig, codegen


@overload_attribute(USMNdArray, "sycl_queue", target=DPEX_TARGET_NAME)
def dpnp_nd_array_sycl_queue(arr):
    """Returns :class:`dpctl.SyclQueue` object associated with USM data.

    This is an overloaded attribute implementation for dpnp.sycl_queue.

    Args:
        arr (numba_dpex.core.types.DpnpNdArray): Input array from which to
            take sycl_queue.

    Returns:
        function: Local function `ol_dpnp_nd_array_sycl_queue()`.
    """

    def get(arr):
        return ol_usm_nd_array_sycl_queue(arr)

    return get
