import dpnp
import numpy as np

import numba_dpex as dpex


@dpex.kernel
def vecadd(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = np.add(a[i], b[i])


a = dpnp.ones(1024)
b = dpnp.ones(1024)
c = dpnp.zeros(1024)

print("Before vecadd...")
print(c)
vecadd[
    1024,
](a, b, c)
print("After vecadd...")
print(c)
