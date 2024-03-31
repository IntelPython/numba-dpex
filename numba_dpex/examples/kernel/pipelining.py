# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The overlapping of memory copy and computation depends on both the complexity of
the computation and the device hardware specifications such as memory bandwidth.
The example serves as illustration of how memory copy and compute can be
overlapped using numba_dpex's async kernel execution feature. For real world use
cases, the compute kernel will need proper profiling and the pipelining of the
compute and memory copy kernels will need to be tailored per device
capabilities.
"""

import argparse
import time

import dpctl
import dpnp
import numpy as np

import numba_dpex as dpex


@dpex.kernel
def async_kernel(item, x):
    idx = item.get_id(0)

    for i in range(1300):
        den = x.dtype.type(i + 1)
        x[idx] += x.dtype.type(1) / (den * den * den)


def run_serial(host_arr, n_itr):
    t0 = time.time()
    q = dpctl.SyclQueue()

    a_host = dpnp.asarray(host_arr, usm_type="host", sycl_queue=q)
    usm_host_data = dpnp.get_usm_ndarray(a_host).usm_data

    batch_shape = (n_itr,) + a_host.shape
    device_alloc = dpnp.empty(batch_shape, usm_type="device", sycl_queue=q)

    for offset in range(n_itr):
        _a = device_alloc[offset]
        _a_data = dpnp.get_usm_ndarray(_a).usm_data

        q.memcpy(_a_data, usm_host_data, usm_host_data.nbytes)

        dpex.call_kernel(
            async_kernel,
            dpex.Range(len(_a)),
            _a,
        )

    dt = time.time() - t0

    return dt, None, None


def run_pipeline(host_arr, n_itr):
    t0 = time.time()
    q = dpctl.SyclQueue()

    a_host = dpnp.asarray(host_arr, usm_type="host", sycl_queue=q)
    usm_host_data = dpnp.get_usm_ndarray(a_host).usm_data

    batch_shape = (n_itr,) + a_host.shape
    device_alloc = dpnp.empty(batch_shape, usm_type="device", sycl_queue=q)

    e_a = None
    e_b = None

    for offset in range(n_itr):
        _a = device_alloc[offset]
        _a_data = dpnp.get_usm_ndarray(_a).usm_data

        e_a = q.memcpy_async(
            _a_data,
            usm_host_data,
            usm_host_data.nbytes,
            [e_a] if e_a is not None else [],
        )

        e_a.wait()

        _, e_a = dpex.call_kernel_async(
            async_kernel,
            dpex.Range(len(_a)),
            (e_a,),
            _a,
        )

        e_a.wait()

        e_a, e_b = e_b, e_a

    q.wait()
    dt = time.time() - t0

    return dt, None, None


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--n",
        type=int,
        default=2_000_000,
        help="an integer for the input array",
    )
    parser.add_argument(
        "--n_itr", type=int, default=100, help="number of iterations"
    )
    parser.add_argument("--reps", type=int, default=5, help="number of repeats")
    parser.add_argument(
        "--algo",
        type=str,
        default="pipeline",
        choices=["pipeline", "serial"],
        help="algo",
    )

    args = parser.parse_args()

    print(
        "timing %d elements for %d iterations" % (args.n, args.n_itr),
        flush=True,
    )

    print("using %f MB of memory" % (args.n * 4 / 1024 / 1024), flush=True)

    a = np.arange(args.n, dtype=np.float32)

    algo_func = {
        "pipeline": run_pipeline,
        "serial": run_serial,
    }.get(args.algo)

    for _ in range(args.reps):
        dtp = algo_func(a, args.n_itr)
        print(f"{args.algo} time tot|pci|cmp|speedup: {dtp}", flush=True)


if __name__ == "__main__":
    main()
