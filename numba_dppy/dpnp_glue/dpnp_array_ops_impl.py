import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba import types
from numba.core.typing import signature
from . import stubs
import numba_dppy.dpnp_glue as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np
from numba_dppy import dpctl_functions
import numba_dppy


# @register_jitable
# def common_impl(a, out, dpnp_func, print_debug):
#     if a.size == 0:
#         raise ValueError("Passed Empty array")

#     sycl_queue = dpctl_functions.get_current_queue()
#     a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
#     dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

#     out_usm = dpctl_functions.malloc_shared(a.itemsize, sycl_queue)
#     print(str(dpnp_func))
#     dpnp_func(a_usm, out_usm, a.size)

#     dpctl_functions.queue_memcpy(
#         sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
#     )

#     dpctl_functions.free_with_queue(a_usm, sycl_queue)
#     dpctl_functions.free_with_queue(out_usm, sycl_queue)

#     dpnp_ext._dummy_liveness_func([out.size])

#     if print_debug:
#         print("dpnp implementation")


# @overload(stubs.dpnp.sort)
# def dpnp_sort_impl(a):
#     name = "sort"
#     dpnp_lowering.ensure_dpnp(name)

#     ret_type = types.void
#     """
#     dpnp source:
#     https://github.com/IntelPython/dpnp/blob/master/dpnp/backend/kernels/dpnp_krnl_sorting.cpp#L90

#     Function declaration:
#     void dpnp_sort_c(void* array1_in, void* result1, size_t size)

#     """
#     sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
#     dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

#     res_dtype = np.int64

#     def dpnp_impl(a):
#         if a.size == 0:
#             raise ValueError("Passed Empty array")

#         sycl_queue = dpctl_functions.get_current_queue()

#         a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
#         dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

#         out = np.arange(a.size, dtype=res_dtype)
#         out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

#         dpnp_func(a_usm, out_usm, a.size)

#         dpctl_functions.queue_memcpy(
#             sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
#         )

#         dpctl_functions.free_with_queue(a_usm, sycl_queue)
#         dpctl_functions.free_with_queue(out_usm, sycl_queue)

#         dpnp_ext._dummy_liveness_func([a.size, out.size])

#         return out

#     return dpnp_impl


# @overload(stubs.dpnp.sort)
# def dpnp_sort_impl(a):
#     name = "sort"
#     dpnp_lowering.ensure_dpnp(name)

#     ret_type = types.void
#     """
#     dpnp source:
#     https://github.com/IntelPython/dpnp/blob/master/dpnp/backend/kernels/dpnp_krnl_sorting.cpp#L90

#     Function declaration:
#     void dpnp_sort_c(void* array1_in, void* result1, size_t size)

#     """
#     sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
#     dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

#     PRINT_DEBUG = dpnp_lowering.DEBUG
#     print("LALAL TYPING")

#     def dpnp_impl(a):
#         out = np.empty(1, dtype=a.dtype)
#         common_impl(a, out, dpnp_func, PRINT_DEBUG)
#         print("LALAL IMPL")
#         return out[0]

#     return dpnp_impl


@overload(stubs.dpnp.sort)
def dpnp_sort_impl(a):
    print("LALALA")
    name = "sort"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/master/dpnp/backend/kernels/dpnp_krnl_sorting.cpp#L90

    Function declaration:
    void dpnp_sort_c(void* array1_in, void* result1, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    res_dtype = a.dtype
    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.arange(a.size, dtype=res_dtype)
        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.size)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        if PRINT_DEBUG:
            print("dpnp implementation")
        return out

    return dpnp_impl
