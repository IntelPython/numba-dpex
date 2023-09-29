# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba_dpex import NdRange, Range, dpjit

ranges = [(10,), (10, 10), (10, 10, 10)]


@pytest.mark.parametrize("r", ranges)
def test_range_unbox_box(r):
    @dpjit
    def _tester(r):
        return r

    r_in = Range(*r)
    r_out = _tester(r_in)

    assert r_out.ndim == r_in.ndim
    assert r_out.dim0 == r_in.dim0
    assert r_out.dim1 == r_in.dim1
    assert r_out.dim2 == r_in.dim2


@pytest.mark.parametrize("r", ranges)
def test_ndrange_unbox_box(r):
    @dpjit
    def _tester(r):
        return r

    gr = lr = Range(*r)
    r_in = NdRange(gr, lr)
    r_out = _tester(r_in)

    assert r_out.global_range.ndim == r_in.global_range.ndim
    assert r_out.local_range.ndim == r_in.local_range.ndim
    assert r_out.global_range.dim0 == r_in.global_range.dim0
    assert r_out.global_range.dim1 == r_in.global_range.dim1
    assert r_out.global_range.dim2 == r_in.global_range.dim2
    assert r_out.local_range.dim0 == r_in.local_range.dim0
    assert r_out.local_range.dim1 == r_in.local_range.dim1
    assert r_out.local_range.dim2 == r_in.local_range.dim2
