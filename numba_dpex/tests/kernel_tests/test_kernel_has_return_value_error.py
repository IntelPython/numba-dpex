# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import filter_strings


def f(a):
    return a


list_of_sig = [
    None,
    ("int32[::1](int32[::1])"),
]


@pytest.fixture(params=list_of_sig)
def sig(request):
    return request.param


@pytest.mark.parametrize("filter_str", filter_strings)
def test_return(filter_str, sig):
    a = np.array(np.random.random(122), np.int32)

    with pytest.raises(dpex.core.exceptions.KernelHasReturnValueError):
        kernel = dpex.kernel(sig)(f)

        device = dpctl.SyclDevice(filter_str)
        with dpctl.device_context(device):
            kernel[a.size, dpex.DEFAULT_LOCAL_SIZE](a)
