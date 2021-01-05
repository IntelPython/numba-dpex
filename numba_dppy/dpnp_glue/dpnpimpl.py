from numba.core.imputils import lower_builtin
from numba.core import types
from numba.core.extending import overload, register_jitable
import numpy as np
from llvmlite import ir
from numba.core.imputils import lower_getattr

from numba.core.typing import signature
from . import stubs
from numba_dppy import dpctl_functions

ll_void_p = ir.IntType(8).as_pointer()


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


def dpnp_func(fn_name, type_names, sig):
    f_ptr = get_dpnp_fptr(fn_name, type_names)

    def get_pointer(obj):
        return f_ptr

    return types.ExternalFunctionPointer(sig, get_pointer=get_pointer)


"""
This function retrieves the pointer to the structure where the shape
of an ndarray is stored. We cast it to void * to make it easier to
pass around.
"""
@lower_getattr(types.Array, "shapeptr")
def array_shape(context, builder, typ, value):
    shape_ptr = builder.gep(
        value.operands[0],
        [context.get_constant(types.int32, 0), context.get_constant(types.int32, 5)],
    )

    return builder.bitcast(shape_ptr, ll_void_p)


@overload(stubs.dpnp.convert_ndarray_to_usm)
def dpnp_convert_ndarray_to_usm_impl(a):
    name = "convert_ndarray_to_usm"

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        out = a.copy()
        _dummy_liveness_func([a.size, out.size])

        print("WAS called")

        return out

    return dpnp_impl
