# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest
from numba.misc.special import typeof

from numba_dpex.core.types import USMNdArray
from numba_dpex.tests._helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)

list_of_dtypes = [
    np.int32,
    np.float32,
    np.int64,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def dtype(request):
    return request.param


list_of_usm_type = [
    "shared",
    "device",
    "host",
]


@pytest.fixture(params=list_of_usm_type)
def usm_type(request):
    return request.param


def test_usm_ndarray_type(dtype, usm_type):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    a = np.array(np.random.random(10), dtype)
    da = dpt.usm_ndarray(a.shape, dtype=a.dtype, buffer=usm_type)

    assert isinstance(typeof(da), USMNdArray)
    assert da.usm_type == usm_type
