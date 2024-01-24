# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp as np

import numba_dpex as ndpx


@ndpx.kernel(debug=True)
def data_parallel_sum(a, b, c):
    i = ndpx.get_global_id(0)
    c[i] = a[i] + b[i]  # Condition breakpoint location


global_size = 10
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

data_parallel_sum[ndpx.Range(global_size)](a, b, c)

print("Done...")
