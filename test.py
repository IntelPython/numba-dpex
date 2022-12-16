import dpctl
import dpnp
from numba import njit

import numba_dpex

a = dpnp.arange(1024)
b = dpnp.arange(1024)


@njit
def foo(a, b):
    c = a + b
    return c


with dpctl.device_context(a.sycl_device):
    c = foo(a, b)
