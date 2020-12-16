import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba import types
from numba.core.typing import signature
from . import stubs
import numba_dppy.experimental_numpy_lowering_overload as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np

class _DPNP_TRANSCENDENTALS_EXTENSION:
    @classmethod
    def dpnp_sum(cls, fn_name, type_names):
        ret_type = types.void
        sig = signature(ret_type, types.voidptr, types.voidptr, types.int64)
        f_ptr = dpnp_ext.get_dpnp_fptr(fn_name, type_names)

        def get_pointer(obj):
            return f_ptr

        return types.ExternalFunctionPointer(sig, get_pointer=get_pointer)


@overload(stubs.dpnp.sum)
def dpnp_sum_impl(a):
    dpnp_lowering.ensure_dpnp("sum")
    dpnp_extension = _DPNP_TRANSCENDENTALS_EXTENSION()
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

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

        dpnp_ext._dummy_liveness_func([out.size])

        return out[0]

    return dpnp_sum_impl
