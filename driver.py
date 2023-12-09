import dpctl
import dpnp

import numba_dpex.experimental as nd_exp
from numba_dpex import Range


@nd_exp.kernel
def test_barriers(a, b):
    nd_exp.group_barrier()

    nd_exp.group_barrier(nd_exp.MemoryScope.work_group)

    nd_exp.sub_group_barrier()

    nd_exp.sub_group_barrier(MemoryScope.sub_group)


q = dpctl.SyclQueue()
a = dpnp.ones(10, sycl_queue=q, dtype=dpnp.int64)
b = dpnp.array(0, sycl_queue=q, dtype=dpnp.int64)

nd_exp.call_kernel(test_barriers, Range(10), a, b)
print(b)
