# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dpnp import ndarray as dpnp_ndarray


@pytest.mark.xfail(reason="dpnp.ndarray does not support flags yet")
def test_dpnp_ndarray_flags():
    assert hasattr(dpnp_ndarray([1]), "flags")
