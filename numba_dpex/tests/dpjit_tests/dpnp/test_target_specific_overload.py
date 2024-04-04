# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests if dpnp dpex specific overloads are not available at numba njit.
"""

import dpnp
import pytest
from numba import njit
from numba.core import errors

from numba_dpex import dpjit


@pytest.mark.parametrize("func", [dpnp.empty, dpnp.ones, dpnp.zeros])
def test_dpnp_dpex_target(func):
    def dpnp_func():
        func(10)

    dpnp_func_njit = njit(dpnp_func)
    dpnp_func_dpjit = dpjit(dpnp_func)

    dpnp_func_dpjit()
    with pytest.raises((errors.TypingError, errors.UnsupportedError)):
        dpnp_func_njit()
