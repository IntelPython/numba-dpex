from numba import types
from numba.core.typing import signature


def dpctl_get_current_queue():
    ret_type = types.voidptr
    sig = signature(ret_type)
    return types.ExternalFunction("DPCTLQueueMgr_GetCurrentQueue", sig)


def dpctl_malloc_shared():
    ret_type = types.voidptr
    sig = signature(ret_type, types.int64, types.voidptr)
    return types.ExternalFunction("DPCTLmalloc_shared", sig)


def dpctl_queue_memcpy():
    ret_type = types.void
    sig = signature(ret_type, types.voidptr, types.voidptr, types.voidptr, types.int64)
    return types.ExternalFunction("DPCTLQueue_Memcpy", sig)


def dpctl_free_with_queue():
    ret_type = types.void
    sig = signature(ret_type, types.voidptr, types.voidptr)
    return types.ExternalFunction("DPCTLfree_with_queue", sig)


get_current_queue = dpctl_get_current_queue()
malloc_shared = dpctl_malloc_shared()
queue_memcpy = dpctl_queue_memcpy()
free_with_queue = dpctl_free_with_queue()
