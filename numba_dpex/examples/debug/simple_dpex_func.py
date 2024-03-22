# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpx


@ndpx.device_func(debug=True)
def func_sum(a_in_func, b_in_func):
    result = a_in_func + b_in_func  # breakpoint location
    return result


@ndpx.kernel(debug=True)
def kernel_sum(item, a_in_kernel, b_in_kernel, c_in_kernel):
    i = item.get_id(0)
    c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])


global_size = 10
a = np.arange(global_size, dtype=np.float32)
b = np.arange(global_size, dtype=np.float32)
c = np.empty_like(a)

ndpx.call_kernel(kernel_sum, ndpx.Range(global_size), a, b, c)

print("Done...")
