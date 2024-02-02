# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import pytest

import numba_dpex as dpex
from numba_dpex.core.exceptions import UnknownGlobalRangeError
from numba_dpex.kernel_api import NdRange, Range


@dpex.kernel
def vecadd(a, b, c):
    i = dpex.get_global_id(0)
    a[i] = b[i] + c[i]


def test_1D_global_range_as_one_tuple():
    k = vecadd[Range(10)]
    assert k._global_range == [10]
    assert k._local_range is None


def test_2D_global_range_and_2D_local_range4():
    k = vecadd[NdRange((10, 10), (10, 10))]
    assert k._global_range == [10, 10]
    assert k._local_range == [10, 10]


def test_unknown_global_range_error():
    device = dpctl.select_default_device()
    a = dpt.ones(10, dtype=dpt.int16, device=device)
    b = dpt.ones(10, dtype=dpt.int16, device=device)
    c = dpt.zeros(10, dtype=dpt.int16, device=device)
    try:
        vecadd(a, b, c)
    except UnknownGlobalRangeError as e:
        assert "No global range" in e.message


if __name__ == "__main__":
    test_unknown_global_range_error()
