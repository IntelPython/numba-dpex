from numba.core.imputils import (lower_builtin)
import numba.dppl.experimental_numpy_lowering_overload as dpnp_lowering
from numba import types
from . import stubs


@lower_builtin(stubs.dpnp.sum, types.Array)
def array_sum(context, builder, sig, args):
    dpnp_lowering.ensure_dpnp("sum")
    return dpnp_lowering.common_sum_prod_impl(context, builder, sig, args, "dpnp_sum")


