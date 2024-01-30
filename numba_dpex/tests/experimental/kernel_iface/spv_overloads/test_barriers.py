import dpnp

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.experimental.kernel_iface import group_barrier


def test_group_barrier():
    """A test for group_barrier function."""

    @dpex_exp.kernel
    def _kernel(a, N):
        i = dpex.get_global_id(0)

        a[i] += 1
        group_barrier()

        if i == 0:
            for idx in range(1, N):
                a[0] += a[idx]

    N = 8196
    a = dpnp.ones(N, dtype=dpnp.int32)
    b = dpnp.ones(N, dtype=dpnp.int32)

    dpex_exp.call_kernel(_kernel, dpex.Range(N), a, N)

    assert a[0] == N * 2
