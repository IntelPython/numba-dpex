# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as dpex


@dpex.kernel(debug=True)
def data_parallel_sum(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]  # Condition breakpoint location


global_size = 10
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

device = dpctl.select_default_device()
with dpctl.device_context(device):
    data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

print("Done...")
