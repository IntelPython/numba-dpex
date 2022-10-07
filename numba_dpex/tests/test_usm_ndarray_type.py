# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import numpy as np
import pytest
from numba.misc.special import typeof

from numba_dpex.dpctl_iface import USMNdArrayType
from numba_dpex.tests._helper import filter_strings

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


@pytest.mark.parametrize("filter_str", filter_strings)
def test_usm_ndarray_type(filter_str, dtype, usm_type):
    a = np.array(np.random.random(10), dtype)
    da = dpt.usm_ndarray(a.shape, dtype=a.dtype, buffer=usm_type)

    assert isinstance(typeof(da), USMNdArrayType)
    assert da.usm_type == usm_type
