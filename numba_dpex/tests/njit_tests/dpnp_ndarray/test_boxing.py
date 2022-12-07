# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for boxing for dpnp.ndarray
"""

import pytest
from dpnp import ndarray as dpnp_ndarray
from numba import njit

dpnp_mark = pytest.mark.xfail(raises=TypeError, reason="No unboxing")


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(dpnp_ndarray([1]), marks=dpnp_mark),
    ],
)
def test_njit(array):
    @njit
    def func(a):
        return a

    func(array)
