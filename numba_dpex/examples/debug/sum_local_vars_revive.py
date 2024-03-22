# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpx


@ndpx.device_func
def revive(x):
    return x


@ndpx.kernel(debug=True)
def data_parallel_sum(item, a, b, c):
    i = item.get_id(0)
    l1 = a[i] + 2.5
    l2 = b[i] * 0.3
    c[i] = l1 + l2
    revive(a)  # pass variable to dummy function


global_size = 10
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

ndpx.call_kernel(data_parallel_sum, ndpx.Range(global_size), a, b, c)

print("Done...")
