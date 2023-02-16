import dpnp
import numpy as np

import numba_dpex as dpex


@dpex.dpjit
def vecadd(a, b):
    return a + b


# @dpex.dpjit
# def sin(a):
#     return np.sin(a)


a = dpnp.ones(10)
b = dpnp.ones(10)
c = vecadd(a, b)

# a = sin(a)
