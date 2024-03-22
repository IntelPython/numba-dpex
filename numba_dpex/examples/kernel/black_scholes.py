# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache 2.0

"""The example shows the implementation of the Black-Scholes formula as a range kernel.

The Black-Scholes model is a mathematical model for derivatives trading with
various underlying assumptions. The example shown here is a simplified
representation of the actual model.

Refer: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model
"""

from math import erf, exp, log, sqrt

import dpnp as np

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

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
    call = np.empty(NOPT)
    put = np.empty(NOPT)

    return price, strike, t, rate, volatility, call, put


@dpex.kernel
def kernel_black_scholes(
    item: kapi.Item, price, strike, t, rate, volatility, call, put
):
    """A range kernel implementing a simplified Black-Scholes model."""
    # Scalars
    mr = -rate
    sig_sig_two = volatility * volatility * 2.0

    # Current index
    i = item.get_id(0)

    # Get inputs into private memory
    p = price[i]
    s = strike[i]
    tt = t[i]

    a = log(p / s)
    b = tt * mr

    z = tt * sig_sig_two
    c = 0.25 * z
    y = 1.0 / sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * erf(w1)
    d2 = 0.5 + 0.5 * erf(w2)

    se = exp(b) * s

    r = p * d1 - se * d2

    # Write back results
    call[i] = r
    put[i] = r - p + se


def main():
    price, strike, t, rate, volatility, call, put = initialize()

    print("Executing on device:")
    price.device.print_device_info()

    dpex.call_kernel(
        kernel_black_scholes,
        dpex.Range(NOPT),
        price,
        strike,
        t,
        rate,
        volatility,
        call,
        put,
    )

    print("Call:", call)
    print("Put:", put)
    print("Done...")


if __name__ == "__main__":
    main()
