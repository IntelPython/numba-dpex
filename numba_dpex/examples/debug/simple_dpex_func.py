# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as dpex


@dpex.func(debug=True)
def func_sum(a_in_func, b_in_func):
    result = a_in_func + b_in_func
    return result


@dpex.kernel(debug=True)
def kernel_sum(a_in_kernel, b_in_kernel, c_in_kernel):
    i = dpex.get_global_id(0)
    c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])


global_size = 10
a = np.arange(global_size, dtype=np.float32)
b = np.arange(global_size, dtype=np.float32)
c = np.empty_like(a)

device = dpctl.select_default_device()
with dpctl.device_context(device):
    kernel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

print("Done...")
