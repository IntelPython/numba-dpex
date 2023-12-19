# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import dpnp
import numpy

import numba_dpex as dpex


@dpex.kernel
def kernel_with_private_memory_allocation(A):
    i = dpex.get_global_id(0)
    prvt_mem = dpex.private.array(shape=10, dtype=dpnp.float32)
    # RangeIter object raises an exception, therefore we can't use it
    # in kernel. TODO: Monkeypatch RangeIter and turn off the raise.
    # See https://github.com/numba/numba/blob/main/numba/cpython/rangeobj.py#L130
    # for idx in range(10):
    #     prvt_mem[idx] = idx * idx # noqa: E800
    idx = 0
    while idx < 10:
        prvt_mem[idx] = idx * idx
        idx += 1
    dpex.barrier(dpex.LOCAL_MEM_FENCE)  # local mem fence
    # RangeIter object raises an exception, therefore we can't use it
    # in kernel. TODO: Monkeypatch RangeIter and turn off the raise.
    # See https://github.com/numba/numba/blob/main/numba/cpython/rangeobj.py#L130
    # for idx in range(10):
    #     A[i] += prvt_mem[idx] # noqa: E800
    idx = 0
    while idx < 10:
        A[i] += prvt_mem[idx]
        idx += 1


def test_private_memory_allocation():
    N = 64
    arr = dpnp.zeros(N, dtype=dpnp.float32)
    kernel_with_private_memory_allocation[dpex.Range(N)](arr)

    nparr = dpnp.asnumpy(arr)

    expected = 0
    for i in range(10):
        expected += i * i

    assert numpy.all(numpy.isclose(nparr, expected))
