import dpctl
import dpnp
import numba as nb
import numpy as np

import numba_dpex as dpex


@dpex.dpjit
def vecadd(a, b):
    c = a + b
    d = dpnp.ones_like(c)
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


a = dpnp.ones(10, device="cpu")
b = dpnp.ones(10, device="cpu")

c = vecadd(a, b)
# s = vecadd_prange(a, b)

# a = sin(a)

# with dpctl.device_context("gpu"):
#     c = vecadd(a, b)
