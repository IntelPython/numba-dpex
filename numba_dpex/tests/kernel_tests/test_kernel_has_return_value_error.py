# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
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
    a = dpnp.arange(1024, dtype=dpnp.int32)

    with pytest.raises(dpex.core.exceptions.KernelHasReturnValueError):
        kernel_fn = dpex.kernel(sig)(f)
        dpex.call_kernel(kernel_fn, dpex.Range(a.size), a)
