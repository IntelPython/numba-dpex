import dpctl
import dpnp
import numba as nb
import numpy as np

import numba_dpex as dpex


@dpex.dpjit
def vecadd(a, s):
    if s == 1:
        d = a + a
    else:
        d = a - a
    return d


@dpex.dpjit
def vecadd_prange(a, b):
    c = a + b
    s = 0
    for i in nb.prange(len(c)):
        s += c[i]
    return s


# @dpex.dpjit
# def sin(a):
#     return np.sin(a)

a = dpnp.ones(1024, dtype=dpnp.int64, usm_type="shared", device="gpu")
b = dpnp.ones(1024, dtype=dpnp.int64, usm_type="shared", device="cpu")

print(a)
c = vecadd(a, 1)

print(c)
print(c.usm_type)
# s = vecadd_prange(a, b)

# a = sin(a)

# with dpctl.device_context("gpu"):
#     c = vecadd(a, b)
