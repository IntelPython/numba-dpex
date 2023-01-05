#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
import dpctl.tensor as dpt

import numba_dpex as dpex
from numba_dpex import usm_ndarray
from numba_dpex.core.kernel_interface.dispatcher import JitKernel

arrty = usm_ndarray(int, 1, "C", "device", "gpu")


@dpex.kernel((arrty, arrty, arrty))
def data_parallel_sum(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


def main():
    a = dpt.arange(0, 100, device="level_zero:gpu:0")
    b = dpt.arange(0, 100, device="level_zero:gpu:0")
    c = dpt.zeros_like(a, device="level_zero:gpu:0")

    # d = Dispatcher(pyfunc=data_parallel_sum)
    # d(a, b, c, global_range=[100])
    data_parallel_sum[(100,)](a, b, c)
    print(dpt.asnumpy(a))
    print(dpt.asnumpy(b))
    print(dpt.asnumpy(c))
    print("Done...")


if __name__ == "__main__":
    main()
