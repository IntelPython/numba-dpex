# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy
import pytest

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

N = 1024


@dpex.kernel
def vecadd_kernel(item: kapi.Item, a, b, c):
    i = item.get_id(0)
    c[i] = a[i] + b[i]


def test_passing_numpy_arrays_as_kernel_args():
    """
    Negative test to verify that NumPy arrays cannot be passed to a kernel.
    """
    a = numpy.ones(N)
    b = numpy.ones(N)
    c = numpy.zeros(N)

    with pytest.raises(Exception):
        dpex.call_kernel(vecadd_kernel, dpex.Range(N), a, b, c)
