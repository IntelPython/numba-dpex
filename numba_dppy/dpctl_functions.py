from numba import types
from numba.core.typing import signature


class _DPCTL_FUNCTIONS:
    @classmethod
    def dpctl_get_current_queue(cls):
        ret_type = types.voidptr
        sig = signature(ret_type)
        return types.ExternalFunction("DPCTLQueueMgr_GetCurrentQueue", sig)

    @classmethod
    def dpctl_malloc_shared(cls):
        ret_type = types.voidptr
        sig = signature(ret_type, types.int64, types.voidptr)
        return types.ExternalFunction("DPCTLmalloc_shared", sig)

    @classmethod
    def dpctl_queue_memcpy(cls):
        ret_type = types.void
        sig = signature(
            ret_type, types.voidptr, types.voidptr, types.voidptr, types.int64
        )
        return types.ExternalFunction("DPCTLQueue_Memcpy", sig)

    @classmethod
    def dpctl_free_with_queue(cls):
        ret_type = types.void
        sig = signature(ret_type, types.voidptr, types.voidptr)
        return types.ExternalFunction("DPCTLfree_with_queue", sig)
