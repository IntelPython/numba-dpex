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

    assert r_out == r_in


@pytest.mark.parametrize("r", ranges)
def test_ndrange_unbox_box(r):
    @dpjit
    def _tester(r):
        return r

    gr = lr = Range(*r)
    r_in = NdRange(gr, lr)
    r_out = _tester(r_in)

    assert r_out == r_in
