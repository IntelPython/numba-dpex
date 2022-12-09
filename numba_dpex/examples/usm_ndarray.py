# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import numpy.testing as testing

import numba_dpex as dpex


@dpex.kernel
def vector_add(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


def driver(a, b, c, global_size):
    vector_add[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)
    npa = dpt.asnumpy(a)
    npb = dpt.asnumpy(b)
    npc = dpt.asnumpy(c)
    testing.assert_equal(npc, npa + npb)


def main():
    N = 1024
    print("N", N)

    a = dpt.arange(N)
    b = dpt.arange(N)
    c = dpt.zeros(N)

    print("Using device ...")
    a.sycl_device.print_device_info()

    driver(a, b, c, N)

    print("Done...")


if __name__ == "__main__":
    main()
