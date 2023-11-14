import dpctl
import dpnp

import numba_dpex.experimental as nd_exp
from numba_dpex import Range, dpjit

print(type(nd_exp.MemoryOrder))
print(type(nd_exp.MemoryOrder.RELAXED))


@nd_exp.kernel
def test_atomic_ref(a, b):
    a[0] = nd_exp.MemoryOrder.RELAXED
    v = nd_exp.AtomicRef(
        b,
        nd_exp.MemoryOrder.RELAXED,
        nd_exp.MemoryScope.DEVICE,
        nd_exp.AddressSpace.GLOBAL,
    )
    v.store(a[0])


q = dpctl.SyclQueue()
a = dpnp.ones(10, sycl_queue=q, dtype=dpnp.int64)
b = dpnp.array(0, sycl_queue=q, dtype=dpnp.int64)

nd_exp.call_kernel(test_atomic_ref, Range(10), a, b)
print(b)
