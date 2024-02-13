# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy
import pytest

from numba_dpex import kernel_api as kapi


def _slm_kernel(nd_item: kapi.NdItem, a, slm):
    i = nd_item.get_global_linear_id()
    j = nd_item.get_local_linear_id()

    slm[j] = 100
    a[i] = slm[i]


def test_local_accessor_data_inaccessible_outside_kernel():
    la = kapi.LocalAccessor((100,), dtype=numpy.float32)

    with pytest.raises(NotImplementedError):
        print(la[0])

    with pytest.raises(NotImplementedError):
        la[0] = 10


def test_local_accessor_use_inside_kernel():

    a = numpy.empty(32)
    slm = kapi.LocalAccessor(32, dtype=a.dtype)

    # launches one work group with 32 work item. Each work item initializes its
    # position in the SLM to 100 and then writes it to the global array `a`.
    kapi.call_kernel(_slm_kernel, kapi.NdRange((32,), (32,)), a, slm)

    assert numpy.all(a == 100)


def test_local_accessor_usage_not_allowed_with_range_kernel():

    a = numpy.empty(32)
    slm = kapi.LocalAccessor(32, dtype=a.dtype)

    with pytest.raises(TypeError):
        kapi.call_kernel(_slm_kernel, kapi.Range((32,)), a, slm)
