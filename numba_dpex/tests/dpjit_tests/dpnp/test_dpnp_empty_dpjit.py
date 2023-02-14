# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp ndarray constructors."""

import dpnp
import pytest

from numba_dpex import dpjit

shapes = [10, (2, 5)]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]
usm_types = ["device", "shared", "host"]
devices = ["cpu", "unknown"]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize("device", devices)
def test_dpnp_empty(shape, dtype, usm_type, device):
    @dpjit
    def func(shape):
        dpnp.empty(shape=shape, dtype=dtype, usm_type=usm_type, device=device)

    @dpjit
    def func1(shape):
        c = dpnp.empty(
            shape=shape, dtype=dtype, usm_type=usm_type, device=device
        )
        return c

    func(shape)

    func1(shape)
