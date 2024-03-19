# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import dpnp
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.kernel_api import (
    LocalAccessor,
    MemoryScope,
    NdItem,
    group_barrier,
)
from numba_dpex.tests._helper import get_all_dtypes

list_of_supported_dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)


@dpex_exp.kernel
def _kernel1(nd_item: NdItem, a, slm):
    i = nd_item.get_global_linear_id()

    # TODO: overload nd_item.get_local_id()
    j = (nd_item.get_local_id(0),)

    slm[j] = 0
    group_barrier(nd_item.get_group(), MemoryScope.WORK_GROUP)

    for m in range(100):
        slm[j] += i * m
        group_barrier(nd_item.get_group(), MemoryScope.WORK_GROUP)

    a[i] = slm[j]


@dpex_exp.kernel
def _kernel2(nd_item: NdItem, a, slm):
    i = nd_item.get_global_linear_id()

    # TODO: overload nd_item.get_local_id()
    j = (nd_item.get_local_id(0), nd_item.get_local_id(1))

    slm[j] = 0
    group_barrier(nd_item.get_group(), MemoryScope.WORK_GROUP)

    for m in range(100):
        slm[j] += i * m
        group_barrier(nd_item.get_group(), MemoryScope.WORK_GROUP)

    a[i] = slm[j]


@dpex_exp.kernel
def _kernel3(nd_item: NdItem, a, slm):
    i = nd_item.get_global_linear_id()

    # TODO: overload nd_item.get_local_id()
    j = (
        nd_item.get_local_id(0),
        nd_item.get_local_id(1),
        nd_item.get_local_id(2),
    )

    slm[j] = 0
    group_barrier(nd_item.get_group(), MemoryScope.WORK_GROUP)

    for m in range(100):
        slm[j] += i * m
        group_barrier(nd_item.get_group(), MemoryScope.WORK_GROUP)

    a[i] = slm[j]


@pytest.mark.parametrize("supported_dtype", list_of_supported_dtypes)
@pytest.mark.parametrize(
    "nd_range, _kernel",
    [
        (dpex.NdRange((32,), (32,)), _kernel1),
        (dpex.NdRange((32, 1), (32, 1)), _kernel2),
        (dpex.NdRange((1, 32, 1), (1, 32, 1)), _kernel3),
    ],
)
def test_local_accessor(supported_dtype, nd_range: dpex.NdRange, _kernel):
    """A test for passing a LocalAccessor object as a kernel argument."""

    N = 32
    a = dpnp.empty(N, dtype=supported_dtype)
    slm = LocalAccessor(nd_range.local_range, dtype=a.dtype)

    # A single work group with 32 work items is launched. Each work item
    # computes the sum of (0..99) * its get_global_linear_id i.e.,
    # `4950 * get_global_linear_id` and stores it into the work groups local
    # memory. The local memory is of size 32*64 elements of the requested dtype.
    # The result is then stored into `a` in global memory
    dpex_exp.call_kernel(_kernel, nd_range, a, slm)

    for idx in range(N):
        assert a[idx] == 4950 * idx


def test_local_accessor_argument_to_range_kernel():
    """Checks if an exception is raised when passing a local accessor to a
    RangeType kernel.
    """
    N = 32
    a = dpnp.empty(N)
    slm = LocalAccessor((32 * 64), dtype=a.dtype)

    # Passing a local_accessor to a RangeType kernel should raise an exception.
    # A TypeError is raised if NUMBA_CAPTURED_ERROR=new_style and a
    # numba.TypingError is raised if NUMBA_CAPTURED_ERROR=old_style
    with pytest.raises((TypeError, TypingError)):
        dpex_exp.call_kernel(_kernel1, dpex.Range(N), a, slm)
