from numba.core.imputils import (lower_builtin)
import numba.dppl.experimental_numpy_lowering_overload as dpnp_lowering
from numba import types
from numba.core.typing import signature
from numba.core.extending import overload
from . import stubs
import numpy as np


def get_dpnp_fptr(fn_name, type_names):
    from . import dpnp_fptr_interface as dpnp_glue
    f_ptr = dpnp_glue.get_dpnp_fn_ptr(fn_name, type_names)
    return f_ptr

class RetrieveDpnpFnPtr(types.ExternalFunctionPointer):
    def __init__(self, fn_name, type_names, sig, get_pointer):
        self.fn_name = fn_name
        self.type_names = type_names
        super(RetrieveDpnpFnPtr, self).__init__(sig, get_pointer)

class _DPNP_EXTENSION:
    def __init__(self, name):
        dpnp_lowering.ensure_dpnp(name)

    @classmethod
    def get_sycl_queue(cls):
        ret_type  = types.voidptr
        sig       = signature(ret_type)
        return types.ExternalFunction("DPPLQueueMgr_GetCurrentQueue", sig)

    @classmethod
    def allocate_usm_shared(cls):
        ret_type  = types.voidptr
        sig       = signature(ret_type, types.int64, types.voidptr)
        return types.ExternalFunction("DPPLmalloc_shared", sig)

    @classmethod
    def copy_usm(cls):
        ret_type  = types.void
        sig       = signature(ret_type, types.voidptr, types.voidptr, types.voidptr, types.int64)
        return types.ExternalFunction("DPPLQueue_Memcpy", sig)

    @classmethod
    def free_usm(cls):
        ret_type  = types.void
        sig       = signature(ret_type, types.voidptr, types.voidptr)
        return types.ExternalFunction("DPPLfree_with_queue", sig)


    @classmethod
    def dpnp_sum(cls, fn_name, type_names):
        ret_type  = types.void
        sig       = signature(ret_type, types.voidptr, types.voidptr, types.int64)
        f_ptr     = get_dpnp_fptr(fn_name, type_names)
        def get_pointer(obj):
            return f_ptr
        return types.ExternalFunctionPointer(sig, get_pointer=get_pointer)


@overload(stubs.dpnp.sum)
def dpnp_sum_impl(a):
    dpnp_extension = _DPNP_EXTENSION("sum")

    dpnp_sum = dpnp_extension.dpnp_sum("dpnp_sum", [a.dtype.name, "NONE"])

    get_sycl_queue = dpnp_extension.get_sycl_queue()
    allocate_usm_shared = dpnp_extension.allocate_usm_shared()
    copy_usm = dpnp_extension.copy_usm()
    free_usm = dpnp_extension.free_usm()

    def dpnp_sum_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()
        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out_usm = allocate_usm_shared(a.itemsize, sycl_queue)

        dpnp_sum(a_usm, out_usm, a.size)

        out = np.empty(1, dtype=a.dtype)
        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        return out[0]

    return dpnp_sum_impl

