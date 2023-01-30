# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as dpex
from numba_dpex.core.kernel_interface.utils import Range


@dpex.func
def revive(x):
    return x


@dpex.kernel(debug=True)
def data_parallel_sum(a, b, c):
    i = dpex.get_global_id(0)
    l1 = a[i] + 2.5
    l2 = b[i] * 0.3
    c[i] = l1 + l2
    revive(a)  # pass variable to dummy function


global_size = 10
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

device = dpctl.select_default_device()
with dpctl.device_context(device):
    data_parallel_sum[Range(global_size)](a, b, c)

print("Done...")
