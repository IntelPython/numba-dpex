# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
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

    assert r_out == r_expected


@pytest.mark.parametrize("r", ranges)
def test_ndrange_unbox_box(r):
    @dpjit
    def _tester(r):
        gr = lr = Range(*r)
        return NdRange(gr, lr)

    gr = lr = Range(*r)
    r_expected = NdRange(gr, lr)
    r_out = _tester(r)

    assert r_out == r_expected
