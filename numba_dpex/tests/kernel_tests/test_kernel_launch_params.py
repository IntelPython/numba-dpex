# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import numba_dpex as dpex
from numba_dpex.core.exceptions import (
    IllegalRangeValueError,
    InvalidKernelLaunchArgsError,
)


@dpex.kernel
def vecadd(a, b, c):
    i = dpex.get_global_id(0)
    a[i] = b[i] + c[i]


def test_1D_global_range_as_int():
    k = vecadd[10]
    assert k._global_range == [10]
    assert k._local_range is None


def test_1D_global_range_as_one_tuple():
    k = vecadd[
        10,
    ]
    assert k._global_range == [10]
    assert k._local_range is None


def test_1D_global_range_as_list():
    k = vecadd[[10]]
    assert k._global_range == [10]
    assert k._local_range is None


@pytest.mark.xfail
def test_1D_global_range_and_1D_local_range():
    k = vecadd[10, 10]
    assert k._global_range == [10]
    assert k._local_range == [10]


def test_1D_global_range_and_1D_local_range2():
    k = vecadd[[10, 10]]
    assert k._global_range == [10]
    assert k._local_range == [10]


def test_1D_global_range_and_1D_local_range3():
    k = vecadd[(10,), (10,)]
    assert k._global_range == [10]
    assert k._local_range == [10]


def test_2D_global_range_and_2D_local_range():
    k = vecadd[(10, 10), (10, 10)]
    assert k._global_range == [10, 10]
    assert k._local_range == [10, 10]


def test_2D_global_range_and_2D_local_range2():
    k = vecadd[[10, 10], (10, 10)]
    assert k._global_range == [10, 10]
    assert k._local_range == [10, 10]


def test_2D_global_range_and_2D_local_range3():
    k = vecadd[(10, 10), [10, 10]]
    assert k._global_range == [10, 10]
    assert k._local_range == [10, 10]


def test_2D_global_range_and_2D_local_range4():
    k = vecadd[[10, 10], [10, 10]]
    assert k._global_range == [10, 10]
    assert k._local_range == [10, 10]


def test_deprecation_warning_for_empty_local_range():
    with pytest.deprecated_call():
        k = vecadd[[10, 10], []]
    assert k._global_range == [10, 10]
    assert k._local_range is None


def test_deprecation_warning_for_empty_local_range2():
    with pytest.deprecated_call():
        k = vecadd[10, []]
    assert k._global_range == [10]
    assert k._local_range is None


def test_illegal_kernel_launch_arg():
    with pytest.raises(InvalidKernelLaunchArgsError):
        vecadd[10, 10, []]


def test_illegal_range_error():
    with pytest.raises(IllegalRangeValueError):
        vecadd[[], []]


def test_illegal_range_error2():
    with pytest.raises(IllegalRangeValueError):
        vecadd[[], 10]


def test_illegal_range_error3():
    with pytest.raises(IllegalRangeValueError):
        vecadd[(), 10]
