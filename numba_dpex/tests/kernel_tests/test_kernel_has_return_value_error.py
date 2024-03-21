# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
from numba_dpex import int32, usm_ndarray
from numba_dpex.core.exceptions import KernelHasReturnValueError
from numba_dpex.core.types.kernel_api.index_space_ids import ItemType

i32arrty = usm_ndarray(ndim=1, dtype=int32, layout="C")
item_ty = ItemType(ndim=1)


def f(item, a):
    return a


list_of_sig = [
    None,
    (i32arrty(item_ty, i32arrty)),
]


@pytest.fixture(params=list_of_sig)
def sig(request):
    return request.param


def test_return(sig):
    a = dpnp.arange(1024, dtype=dpnp.int32)

    with pytest.raises((TypingError, KernelHasReturnValueError)) as excinfo:
        kernel_fn = dpex.kernel(sig)(f)
        dpex.call_kernel(kernel_fn, dpex.Range(a.size), a)

    if isinstance(excinfo.type, TypingError):
        assert "KernelHasReturnValueError" in excinfo.value.args[0]
