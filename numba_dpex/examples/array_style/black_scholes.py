# Copyright 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0


from math import erf, sqrt

import dpnp as np
import numpy.testing as testing

import numba_dpex as ndpx

# Stock price range
S0L = 10.0
S0H = 50.0

# Strike range
XL = 10.0
XH = 50.0

# Maturity range
TL = 1.0
TH = 2.0

# Risk-free rate assumed constant
RISK_FREE = 0.1

# Volatility assumed constants
VOLATILITY = 0.2

# Number of call-put options
NOPT = 1024 * 1024

# Random seed
SEED = 777


def initialize():
    np.random.seed(SEED)
    price = np.random.uniform(S0L, S0H, NOPT)
    strike = np.random.uniform(XL, XH, NOPT)
    t = np.random.uniform(TL, TH, NOPT)
    rate = RISK_FREE
    volatility = VOLATILITY
    return price, strike, t, rate, volatility


# Cumulative normal distribution function
@ndpx.vectorize
def cndf(x):
    one_over_sqrt2 = 1.0 / sqrt(2.0)
    return 0.5 + 0.5 * erf(x * one_over_sqrt2)


# call = cndf(d1)*price - cndf(d2)*strike*exp(-r*t)
# put = strike*exp(-r*t) - price + call
# d1 = (ln(price/strike) + (r + volatility*volatility/2)*t) / (volatility*sqrt(t))
# d2 = d1 - volatility*sqrt(t)
@ndpx.njit
def black_scholes(price, strike, t, rate, volatility):
    log_term = np.log(price / strike)
    rt = rate * t
    sigma_sqrt_t = volatility * np.sqrt(t)
    sigma2_t_over_2 = sigma_sqrt_t * sigma_sqrt_t * 0.5
    exp_mrt = np.exp(-rt)
    strike_exp_mrt = strike * exp_mrt

    d1 = (log_term + (rt + sigma2_t_over_2)) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    call = cndf(d1) * price - cndf(d2) * strike_exp_mrt
    put = strike_exp_mrt - price + call
    return call, put


def main():
    price, strike, t, rate, volatility = initialize()

    print("Using device ...")
    print(price.device)

    call, put = black_scholes(price, strike, t, rate, volatility)

    call_host = np.asnumpy(call)
    put_host = np.asnumpy(put)

    testing.assert_equal(call_host[0], 1.688330986922967)
    testing.assert_equal(put_host[0], 1.1139126320995452)

    print("Done...")


if __name__ == "__main__":
    main()
