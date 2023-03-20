# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex import int32, usm_ndarray

i32arrty = usm_ndarray(ndim=1, dtype=int32, layout="C")


def f(a):
    return a


list_of_sig = [
    None,
    (i32arrty(i32arrty)),
]


@pytest.fixture(params=list_of_sig)
def sig(request):
    return request.param


def test_return(sig):
    a = dpt.arange(1024, dtype=dpt.int32, device="0")

    with pytest.raises(dpex.core.exceptions.KernelHasReturnValueError):
        kernel = dpex.kernel(sig)(f)
        kernel[a.size, dpex.DEFAULT_LOCAL_SIZE](a)
