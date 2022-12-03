#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
import dpctl.tensor as dpt

import numba_dpex as dpex
from numba_dpex.core.kernel_interface.dispatcher import (
    Dispatcher,
    get_ordered_arg_access_types,
)


def test_decorator():
    @dpex.kernel
    def data_parallel_sum(a, b, c):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    a = dpt.arange(0, 100, device="level_zero:gpu:0")
    b = dpt.arange(0, 100, device="level_zero:gpu:0")
    c = dpt.zeros_like(a, device="level_zero:gpu:0")

    data_parallel_sum[(100,)](a, b, c)

    print(dpt.asnumpy(a))
    print(dpt.asnumpy(b))
    print(dpt.asnumpy(c))
    print("Done...")


def test_dispatcher():
    def data_parallel_sum(a, b, c):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    a = dpt.arange(0, 100, device="level_zero:gpu:0")
    b = dpt.arange(0, 100, device="level_zero:gpu:0")
    c = dpt.zeros_like(a, device="level_zero:gpu:0")

    d = Dispatcher(
        data_parallel_sum,
        array_access_specifiers=get_ordered_arg_access_types(
            data_parallel_sum, None
        ),
    )

    for i in range(10):
        print("Dispatch =", i)
        d(a, b, c, global_range=[100])

    print(dpt.asnumpy(a))
    print(dpt.asnumpy(b))
    print(dpt.asnumpy(c))
    print("Done...")

    print("Cache hits =", d.cache_hits)


if __name__ == "__main__":
    test_decorator()
    test_dispatcher()
