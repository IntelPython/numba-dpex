# Copyright 2020, 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dpctl
import numba
import numpy as np
import math
import argparse
import time


@numba.vectorize(nopython=True)
def cndf2(inp):
    out = 0.5 + 0.5 * math.erf((math.sqrt(2.0) / 2.0) * inp)
    return out


@numba.njit(parallel=True, fastmath=True)
def blackscholes(sptprice, strike, rate, volatility, timev):
    """
    A simple implementation of the Black-Scholes formula using the automatic
    offload feature of numba_dppy. In this example, each NumPy array
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
    parser.add_argument("--options", dest="options", type=int, default=10000000)
    args = parser.parse_args()
    options = args.options

    # Run the example of a deafult GPU device
    # Device can be selected using envar SYCL_DEVICE_FILTER.
    # For example:
    #   SYCL_DEVICE_FILTER=opencl:gpu python blacksholes_njit.py
    # Currently, SYCL_DEVICE_FILTER=host is not supported
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        run(10)

    print("Done...")


if __name__ == "__main__":
    main()
