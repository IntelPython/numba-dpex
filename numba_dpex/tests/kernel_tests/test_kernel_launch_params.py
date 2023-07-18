# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import pytest

import numba_dpex as dpex
from numba_dpex.core.exceptions import (
    IllegalRangeValueError,
    InvalidKernelLaunchArgsError,
    UnknownGlobalRangeError,
)
from numba_dpex.core.kernel_interface.indexers import Range


@dpex.kernel
def vecadd(a, b, c):
    i = dpex.get_global_id(0)
    a[i] = b[i] + c[i]


def test_1D_global_range_as_int():
    with pytest.deprecated_call():
        k = vecadd[10]
        assert k._global_range == [10]
        assert k._local_range is None


def test_1D_global_range_as_one_tuple():
    k = vecadd[Range(10)]
    assert k._global_range == [10]
    assert k._local_range is None


def test_1D_global_range_as_list():
    with pytest.deprecated_call():
        k = vecadd[[10]]
        assert k._global_range == [10]
        assert k._local_range is None


def test_1D_global_range_and_1D_local_range1():
    with pytest.deprecated_call():
        k = vecadd[[10, 10]]
        assert k._global_range == [10]
        assert k._local_range == [10]


def test_1D_global_range_and_1D_local_range2():
    with pytest.deprecated_call():
        k = vecadd[(10,), (10,)]
        assert k._global_range == [10]
        assert k._local_range == [10]


def test_2D_global_range_and_2D_local_range1():
    with pytest.deprecated_call():
        k = vecadd[(10, 10), (10, 10)]
        assert k._global_range == [10, 10]
        assert k._local_range == [10, 10]


def test_2D_global_range_and_2D_local_range2():
    with pytest.deprecated_call():
        k = vecadd[[10, 10], (10, 10)]
        assert k._global_range == [10, 10]
        assert k._local_range == [10, 10]


def test_2D_global_range_and_2D_local_range3():
    with pytest.deprecated_call():
        k = vecadd[(10, 10), [10, 10]]
        assert k._global_range == [10, 10]
        assert k._local_range == [10, 10]


def test_2D_global_range_and_2D_local_range4():
    k = vecadd[dpex.NdRange((10, 10), (10, 10))]
    assert k._global_range == [10, 10]
    assert k._local_range == [10, 10]


def test_deprecation_warning_for_empty_local_range1():
    with pytest.deprecated_call():
        k = vecadd[[10, 10], []]
    assert k._global_range == [10, 10]
    assert k._local_range is None


def test_deprecation_warning_for_empty_local_range2():
    with pytest.deprecated_call():
        k = vecadd[10, []]
    assert k._global_range == [10]
    assert k._local_range is None


def test_ambiguous_kernel_launch_params():
    with pytest.deprecated_call():
        k = vecadd[10, 10]
    assert k._global_range == [10]
    assert k._local_range == [10]

    with pytest.deprecated_call():
        k = vecadd[(10, 10)]
    assert k._global_range == [10]
    assert k._local_range == [10]

    with pytest.deprecated_call():
        k = vecadd[((10), (10))]
    assert k._global_range == [10]
    assert k._local_range == [10]


def test_unknown_global_range_error():
    device = dpctl.select_default_device()
    a = dpt.ones(10, dtype=dpt.int16, device=device)
    b = dpt.ones(10, dtype=dpt.int16, device=device)
    c = dpt.zeros(10, dtype=dpt.int16, device=device)
    try:
        vecadd(a, b, c)
    except UnknownGlobalRangeError as e:
        assert "No global range" in e.message


def test_illegal_kernel_launch_arg1():
    with pytest.raises(InvalidKernelLaunchArgsError):
        with pytest.deprecated_call():
            vecadd[()]


def test_illegal_kernel_launch_arg2():
    with pytest.raises(InvalidKernelLaunchArgsError):
        with pytest.deprecated_call():
            vecadd[10, 10, []]


def test_illegal_range_error1():
    with pytest.raises(IllegalRangeValueError):
        with pytest.deprecated_call():
            vecadd[[], []]


def test_illegal_range_error2():
    with pytest.raises(IllegalRangeValueError):
        with pytest.deprecated_call():
            vecadd[[], 10]


def test_illegal_range_error3():
    with pytest.raises(IllegalRangeValueError):
        with pytest.deprecated_call():
            vecadd[(), 10]


if __name__ == "__main__":
    test_unknown_global_range_error()
