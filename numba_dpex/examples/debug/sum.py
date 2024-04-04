# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpx


@ndpx.kernel(debug=True)
def data_parallel_sum(item, a_in_kernel, b_in_kernel, c_in_kernel):
    i = item.get_id(0)  # numba-kernel-breakpoint
    l1 = a_in_kernel[i]  # second-line
    l2 = b_in_kernel[i]  # third-line
    c_in_kernel[i] = l1 + l2  # fourth-line


def driver(a, b, c, global_size):
    print("before : ", a)
    print("before : ", b)
    print("before : ", c)
    ndpx.call_kernel(data_parallel_sum, ndpx.Range(global_size), a, b, c)
    print("after : ", c)


def main():
    global_size = 10
    N = global_size

    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    c = np.empty_like(a)

    driver(a, b, c, global_size)

    print("Done...")


if __name__ == "__main__":
    main()
