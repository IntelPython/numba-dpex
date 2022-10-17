import dpctl
import numpy as np

import numba_dpex as dpex
from numba_dpex import barrier, get_global_id


@dpex.kernel
def data_parallel_sum(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = get_global_id(0)
    c[i] = a[i] + b[i]


device = dpctl.SyclDevice("gpu")


@dpex.kernel
def f(a):
    lm = dpex.local.array(1, np.int32)
    lm[0] = a[0]
    barrier(dpex.CLK_GLOBAL_MEM_FENCE)
    dpex.atomic.add(lm, 0, 1)
    barrier(dpex.CLK_GLOBAL_MEM_FENCE)
    a[0] = lm[0]


a = np.array([0])
print(a)

with dpctl.device_context(device):
    f[100, 100](a)

print(a)

global_size = 10
N = global_size
print("N", N)

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

with dpctl.device_context(device):
    data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

print(c)
