import dpnp
import numba
from numba.core.extending import overload, register_jitable

import numba_dpex as dpex

d = dpnp.arange(10, dtype=dpnp.float32)
out = dpnp.empty(10, dtype=dpnp.float32)
d.sycl_device.print_device_info()
print(d.usm_type)


@dpex.dpjit
def foo(x, out=None):
    return dpnp.cumsum(x, out)
    # return x.cumsum()


print("input before=", d)
e = foo(d, out)
# e = foo(d) # out will be allocated inside overload; returns incorrect result
print(type(e))
print("input after", d)
print("output=", e)
