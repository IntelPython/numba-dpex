# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import argparse

import dpnp
import numba
import numpy as np

import numba_dpex as ndpx


def common_loop_body(param_a, param_b):
    param_c = param_a + numba.float32(10)  # Set breakpoint here
    param_d = param_b * numba.float32(0.5)
    result = param_c + param_d
    return result


def scenario(api):
    print("Using API:", api)

    global_size = 10

    if api == "numba-ndpx-kernel":
        a, b, c = ndpx_arguments(global_size)
        ndpx_func_driver(a, b, c)
    else:
        a, b, c = numba_arguments(global_size)
        numba_func_driver(a, b, c)

    print(a, b, c, sep="\n")


def numba_arguments(N, dtype=np.float32):
    a = np.arange(N, dtype=dtype)
    b = np.arange(N, dtype=dtype)
    c = np.empty_like(a)
    return a, b, c


def ndpx_arguments(N, dtype=dpnp.float32):
    a = dpnp.arange(N, dtype=dtype)
    b = dpnp.arange(N, dtype=dtype)
    c = dpnp.empty_like(a)
    return a, b, c


@numba.njit(debug=True)
def numba_func_driver(a, b, c):
    for i in range(len(c)):
        c[i] = numba_loop_body(a[i], b[i])


def ndpx_func_driver(a, b, c):
    ndpx.call_kernel(_kernel, ndpx.Range(len(c)), a, b, c)


@ndpx.kernel(debug=True)
def _kernel(item, a_in_kernel, b_in_kernel, c_in_kernel):
    i = item.get_id(0)
    c_in_kernel[i] = ndpx_loop_body(a_in_kernel[i], b_in_kernel[i])


numba_loop_body = numba.njit(debug=True)(common_loop_body)
ndpx_loop_body = ndpx.device_func(debug=True)(common_loop_body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        required=False,
        default="numba",
        choices=["numba", "numba-ndpx-kernel"],
        help="Start the version of functions using numba or numba-ndpx API",
    )

    args = parser.parse_args()

    scenario(args.api)

    print("Done...")


if __name__ == "__main__":
    main()
