from numba.core.imputils import lower_builtin
import numba_dppy.experimental_numpy_lowering_overload as dpnp_lowering
from numba import types
from numba.core.typing import signature
from numba.core.extending import overload, register_jitable
from . import stubs
import numpy as np
from numba_dppy.dpctl_functions import _DPCTL_FUNCTIONS


def get_dpnp_fptr(fn_name, type_names):
    from . import dpnp_fptr_interface as dpnp_glue

    f_ptr = dpnp_glue.get_dpnp_fn_ptr(fn_name, type_names)
    return f_ptr


@register_jitable
def _check_finite_matrix(a):
    for v in np.nditer(a):
        if not np.isfinite(v.item()):
            raise np.linalg.LinAlgError("Array must not contain infs or NaNs.")


@register_jitable
def _dummy_liveness_func(a):
    """pass a list of variables to be preserved through dead code elimination"""
    return a[0]


class RetrieveDpnpFnPtr(types.ExternalFunctionPointer):
    def __init__(self, fn_name, type_names, sig, get_pointer):
        self.fn_name = fn_name
        self.type_names = type_names
        super(RetrieveDpnpFnPtr, self).__init__(sig, get_pointer)


class _DPNP_EXTENSION:
    def __init__(self, name):
        dpnp_lowering.ensure_dpnp(name)

    @classmethod
    def dpnp_sum(cls, fn_name, type_names):
        ret_type = types.void
        sig = signature(ret_type, types.voidptr, types.voidptr, types.int64)
        f_ptr = get_dpnp_fptr(fn_name, type_names)

        def get_pointer(obj):
            return f_ptr

        return types.ExternalFunctionPointer(sig, get_pointer=get_pointer)

    @classmethod
    def dpnp_eig(cls, fn_name, type_names):
        ret_type = types.void
        sig = signature(
            ret_type, types.voidptr, types.voidptr, types.voidptr, types.int64
        )
        f_ptr = get_dpnp_fptr(fn_name, type_names)

        def get_pointer(obj):
            return f_ptr

        return types.ExternalFunctionPointer(sig, get_pointer=get_pointer)


@overload(stubs.dpnp.sum)
def dpnp_sum_impl(a):
    dpnp_extension = _DPNP_EXTENSION("sum")
    dpctl_functions = _DPCTL_FUNCTIONS()

    dpnp_sum = dpnp_extension.dpnp_sum("dpnp_sum", [a.dtype.name, "NONE"])

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

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


        _dummy_liveness_func([out.size])

        return out[0]

    return dpnp_sum_impl


@overload(stubs.dpnp.eig)
def dpnp_eig_impl(a):
    dpnp_extension = _DPNP_EXTENSION("eig")
    dpctl_functions = _DPCTL_FUNCTIONS()

    dpnp_eig = dpnp_extension.dpnp_eig("dpnp_eig", [a.dtype.name, "NONE"])

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.float64
    if a.dtype == np.float32:
        res_dtype = np.float32

    def dpnp_eig_impl(a):
        n = a.shape[-1]
        if a.shape[-2] != n:
            msg = "Last 2 dimensions of the array must be square."
            raise ValueError(msg)

        _check_finite_matrix(a)

        wr = np.empty(n, dtype=res_dtype)
        vr = np.empty((n, n), dtype=res_dtype)

        if n == 0:
            return (wr, vr)

        sycl_queue = get_sycl_queue()
        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        wr_usm = allocate_usm_shared(wr.size * wr.itemsize, sycl_queue)
        vr_usm = allocate_usm_shared(vr.size * vr.itemsize, sycl_queue)

        dpnp_eig(a_usm, wr_usm, vr_usm, n)

        copy_usm(sycl_queue, wr.ctypes, wr_usm, wr.size * wr.itemsize)
        copy_usm(sycl_queue, vr.ctypes, vr_usm, vr.size * vr.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(wr_usm, sycl_queue)
        free_usm(vr_usm, sycl_queue)

        _dummy_liveness_func([wr.size, vr.size])

        return (wr, vr)

    return dpnp_eig_impl
