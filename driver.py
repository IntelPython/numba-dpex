import dpnp

import numba_dpex as dpex


@dpex.dpjit
def foo():
    return dpnp.ones(10)


@dpex.dpjit
def bar(a, b):
    return a + b


a = foo()
print(a)
c = bar(a, a)
print(c)
