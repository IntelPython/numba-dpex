# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import time

import dpctl.tensor as dpt
import dpnp
import numba

import numba_dpex as dpex


@dpex.dpjit
def blackscholes(sptprice, strike, timev, rate, volatility):
    """
    A simple implementation of the Black-Scholes formula using the automatic
    offload feature of numba_dpex. In this example, each NumPy array
    expression is identified as a data-parallel kernel and fused together to
    generate a single SYCL kernel. The kernel is automatically offloaded to
    the device specified where the function is invoked.
    """

    a = dpnp.log(sptprice / strike)
    b = timev * -rate
    z = timev * volatility * volatility * 2
    c = 0.25 * z
    y = dpnp.true_divide(1.0, dpnp.sqrt(z))
    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    NofXd1 = 0.5 + 0.5 * dpnp.erf(w1)
    NofXd2 = 0.5 + 0.5 * dpnp.erf(w2)

    futureValue = strike * dpnp.exp(b)
    call = sptprice * NofXd1 - futureValue * NofXd2
    put = call - sptprice + futureValue
    return put


@dpex.dpjit
def init_initStrike(size, initStrike):
    for idx in numba.prange(initStrike.size):
        initStrike[idx] = 40 + (initStrike[idx] + 1.0) / size
    return initStrike


def run(iterations):
    dpt_sptprice = dpt.full((iterations,), 42.0)
    dpt_range_arr = dpt.arange(iterations)
    dpt_full_arr_05 = dpt.full((iterations,), 0.5)
    dpt_volatility = dpt.full((iterations,), 0.2)

    sptprice = dpnp.ndarray(shape=dpt_sptprice.shape, buffer=dpt_sptprice)
    rate = dpnp.ndarray(shape=dpt_full_arr_05.shape, buffer=dpt_full_arr_05)
    volatility = dpnp.ndarray(shape=dpt_volatility.shape, buffer=dpt_volatility)
    timev = dpnp.ndarray(shape=dpt_full_arr_05.shape, buffer=dpt_full_arr_05)
    initStrike = dpnp.ndarray(shape=dpt_range_arr.shape, buffer=dpt_range_arr)
    initStrike = init_initStrike(iterations, initStrike)

    t1 = time.time()
    put = blackscholes(sptprice, initStrike, rate, volatility, timev)
    t = time.time() - t1
    print(put)
    print("SELFTIMED ", t)


def main():
    run(10)


if __name__ == "__main__":
    main()
