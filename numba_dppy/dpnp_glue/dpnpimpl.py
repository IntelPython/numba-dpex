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
