# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import time

import dpctl
import numba
import numpy as np


@numba.vectorize(nopython=True)
def cndf2(inp):
    out = 0.5 + 0.5 * math.erf((math.sqrt(2.0) / 2.0) * inp)
    return out


@numba.njit(parallel=True, fastmath=True)
def blackscholes(sptprice, strike, rate, volatility, timev):
    """
    A simple implementation of the Black-Scholes formula using the automatic
    offload feature of numba_dpex. In this example, each NumPy array
    expression is identified as a data-parallel kernel and fused together to
    generate a single SYCL kernel. The kernel is automatically offloaded to
    the device specified where the function is invoked.
    """
    logterm = np.log(sptprice / strike)
    powterm = 0.5 * volatility * volatility
    den = volatility * np.sqrt(timev)
    d1 = (((rate + powterm) * timev) + logterm) / den
    d2 = d1 - den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike * np.exp(-rate * timev)
    c1 = futureValue * NofXd2
    call = sptprice * NofXd1 - c1
    put = call - futureValue + sptprice
    return put


def run(iterations):
    sptprice = np.full((iterations,), 42.0)
    initStrike = 40 + (np.arange(iterations) + 1.0) / iterations
    rate = np.full((iterations,), 0.5)
    volatility = np.full((iterations,), 0.2)
    timev = np.full((iterations,), 0.5)

    t1 = time.time()
    put = blackscholes(sptprice, initStrike, rate, volatility, timev)
    t = time.time() - t1
    print("checksum: ", sum(put))
    print("SELFTIMED ", t)


def main():
    parser = argparse.ArgumentParser(description="Black-Scholes")
    parser.add_argument("--iter", dest="iter", type=int, default=10)
    args = parser.parse_args()
    iter = args.iter

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        run(iter)

    print("Done...")


if __name__ == "__main__":
    main()
