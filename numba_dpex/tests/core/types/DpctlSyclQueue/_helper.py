from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.extending import intrinsic

from numba_dpex import dpjit


@intrinsic
def _are_queues_equal(typingctx, ty_queue1, ty_queue2):
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
        fnty = llvmir.FunctionType(
            cgutils.bool_t, [cgutils.voidptr_t, cgutils.voidptr_t]
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, "DPCTLQueue_AreEq"
        )
        qref1 = builder.extract_value(args[0], 1)
        qref2 = builder.extract_value(args[1], 1)

        ret = builder.call(fn, [qref1, qref2])

        return ret

    return sig, codegen


@dpjit
def are_queues_equal(queue1, queue2):
    return _are_queues_equal(queue1, queue2)
