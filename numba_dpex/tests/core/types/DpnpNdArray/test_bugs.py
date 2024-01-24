# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dpnp import ndarray as dpnp_ndarray


def test_dpnp_ndarray_flags():
    assert hasattr(dpnp_ndarray([1]), "flags")
