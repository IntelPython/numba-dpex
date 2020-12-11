import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba import types
from numba.core.typing import signature
from . import stubs
import numba_dppy.experimental_numpy_lowering_overload as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np
from numba_dppy.dpctl_functions import _DPCTL_FUNCTIONS

@overload(stubs.dpnp.eig)
def dpnp_eig_impl(a):
    dpnp_lowering.ensure_dpnp("eig")
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L180

    Function declaration:
    void dpnp_eig_c(const void* array_in, void* result1, void* result2, size_t size)

    """
    sig = signature(
        ret_type, types.voidptr, types.voidptr, types.voidptr, types.intp
    )
    dpnp_eig = dpnp_ext.dpnp_func("dpnp_eig", [a.dtype.name, "NONE"], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.float64
    if a.dtype == types.float32:
        res_dtype = np.float32

    def dpnp_eig_impl(a):
        n = a.shape[-1]
        if a.shape[-2] != n:
            msg = "Last 2 dimensions of the array must be square."
            raise ValueError(msg)

        dpnp_ext._check_finite_matrix(a)

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

        dpnp_ext._dummy_liveness_func([wr.size, vr.size])
        print("CCCC")

        return (wr, vr)

    return dpnp_eig_impl
