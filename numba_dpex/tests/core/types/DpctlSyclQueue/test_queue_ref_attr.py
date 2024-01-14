# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.extending import intrinsic

from numba_dpex import dpjit
from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME


@intrinsic(target=DPEX_TARGET_NAME)
def are_queues_equal(typingctx, ty_queue1, ty_queue2):
    """Calls dpctl's libsyclinterface's DPCTLQueue_AreEq to see if two
    dpctl.SyclQueue objects point to the same sycl queue.

    Args:
        typingctx: The typing context used during lowering.
        ty_queue1: Type of the first queue object,
            i.e., numba_dpex.types.DpctlSyclQueue
        ty_queue2: Type of the second queue object,
            i.e., numba_dpex.types.DpctlSyclQueue

    Returns:
        tuple: The signature of the intrinsic function and the codegen function
            to lower the intrinsic.
    """
    result_type = types.boolean
    sig = result_type(ty_queue1, ty_queue2)

    # defines the custom code generation
    def codegen(context, builder, sig, args):
        q1 = cgutils.create_struct_proxy(ty_queue1)(
            context, builder, value=args[0]
        )
        q2 = cgutils.create_struct_proxy(ty_queue2)(
            context, builder, value=args[1]
        )

        fnty = llvmir.FunctionType(
            cgutils.bool_t, [cgutils.voidptr_t, cgutils.voidptr_t]
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, "DPCTLQueue_AreEq"
        )

        ret = builder.call(fn, [q1.queue_ref, q2.queue_ref])

        return ret

    return sig, codegen


def test_queue_ref_access_in_dpjit():
    """Tests if we can access the queue_ref attribute of a dpctl.SyclQueue
    PyObject inside dpjit and pass it to a native C function, in this case
    dpctl's libsyclinterface's DPCTLQueue_AreEq.

    Checks if the result of queue equality check done inside dpjit is the
    same as when done in Python.
    """

    @dpjit
    def test_queue_equality(queue1, queue2):
        return are_queues_equal(queue1, queue2)

    q1 = dpctl.SyclQueue()
    q2 = dpctl.SyclQueue()

    expected = q1 == q2
    actual = test_queue_equality(q1, q2)

    assert expected == actual

    d = dpctl.SyclDevice()
    cq1 = dpctl._sycl_queue_manager.get_device_cached_queue(d)
    cq2 = dpctl._sycl_queue_manager.get_device_cached_queue(d)

    actual = test_queue_equality(cq1, cq2)
    expected = cq1 == cq2

    assert expected == actual
