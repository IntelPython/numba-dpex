# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba_dpex import NdRange, Range, dpjit

ranges = [(10,), (10, 10), (10, 10, 10)]


@pytest.mark.parametrize("r", ranges)
def test_range_ctor(r):
    @dpjit
    def _tester(r):
        return Range(*r)

    r_expected = Range(*r)
    r_out = _tester(r)

    assert r_out.ndim == r_expected.ndim
    assert r_out.dim0 == r_expected.dim0
    assert r_out.dim1 == r_expected.dim1
    assert r_out.dim2 == r_expected.dim2


@pytest.mark.parametrize("r", ranges)
def test_ndrange_unbox_box(r):
    @dpjit
    def _tester(r):
        gr = lr = Range(*r)
        return NdRange(gr, lr)

    gr = lr = Range(*r)
    r_expected = NdRange(gr, lr)
    r_out = _tester(r)

    assert r_out.global_range.ndim == r_expected.global_range.ndim
    assert r_out.local_range.ndim == r_expected.local_range.ndim
    assert r_out.global_range.dim0 == r_expected.global_range.dim0
    assert r_out.global_range.dim1 == r_expected.global_range.dim1
    assert r_out.global_range.dim2 == r_expected.global_range.dim2
    assert r_out.local_range.dim0 == r_expected.local_range.dim0
    assert r_out.local_range.dim1 == r_expected.local_range.dim1
    assert r_out.local_range.dim2 == r_expected.local_range.dim2
