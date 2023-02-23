import dpnp
import numpy as np

import numba_dpex as dpex


@dpex.dpjit
def vecadd(a, b):
    a1 = dpnp.ones(10)
    c = a + b
    # d = dpnp.ones_like(c)
    d = b + 2
    return c + d


# @dpex.dpjit
# def sin(a):
#     return np.sin(a)


a = dpnp.ones(shape=(10, 10), device="cpu")
b = dpnp.ones(shape=10, device="gpu")
c = vecadd(a, b)

# a = sin(a)


# c = a + b

# d = dpnp.ones_like(c)

# {d: ty1, d: ty2}

# -> {d: ty1_new, d: ty2_new}
