# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest
from numba import njit

from numba_dpex.tests._helper import dpnp_debug, filter_strings, skip_no_dpnp

pytestmark = skip_no_dpnp


@pytest.mark.parametrize("filter_str", filter_strings)
@pytest.mark.parametrize(
    "arr",
    [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    ids=["[1, 2, 3, 4]", "[1, 2, 3, 4, 5, 6, 7, 8, 9]"],
)
def test_repeat(filter_str, arr):
    a = np.array(arr)
    repeats = 2

    def fn(a, repeats):
        return np.repeat(a, repeats)

    f = njit(fn)
    with dpctl.device_context(filter_str), dpnp_debug():
        actual = f(a, repeats)

    expected = fn(a, repeats)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
