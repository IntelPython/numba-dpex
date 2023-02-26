import dpctl
import dpnp
import numba as nb
import numpy as np

import numba_dpex as dpex


@dpex.dpjit
def vecadd(a, b):
    return a + b


@dpex.dpjit
def vecadd1(a, b):
    return dpnp.add(a, b)


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

a = dpnp.ones(10)
b = dpnp.ones(10)
# c = vecadd(a, b)

c = vecadd1(a, b)
# s = vecadd_prange(a, b)

# a = sin(a)

# with dpctl.device_context("gpu"):
#     c = vecadd(a, b)
