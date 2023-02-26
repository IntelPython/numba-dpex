import dpnp

import numba_dpex as dpex


@dpex.dpjit
def foo2(a, b):
    return dpnp.add(a, b)


a = dpnp.ones(10)
b = dpnp.ones(10)


f = foo2(a, b)
