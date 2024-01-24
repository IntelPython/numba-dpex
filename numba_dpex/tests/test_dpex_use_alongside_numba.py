# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
This module contains tests to ensure that numba.njit works with numpy after
importing numba_dpex. Aka lazy testing if we break numba's default behavior.
"""

import numba as nb
import numpy as np

import numba_dpex


@nb.njit
def add1(a):
    return a + 1


def add_py(a, b):
    return np.add(a, b)


add_jit = nb.njit(add_py)


def test_add1():
    a = np.asarray([1j], dtype=np.complex64)
    assert np.array_equal(add1(a), np.asarray([1 + 1j], dtype=np.complex64))


def test_add_py():
    a = np.ones((10,), dtype=np.complex128)
    assert np.array_equal(add_py(a, 1.5), np.full((10,), 2.5, dtype=a.dtype))


def test_add_jit():
    a = np.ones((10,), dtype=np.complex128)
    assert np.array_equal(add_jit(a, 1.5), np.full((10,), 2.5, dtype=a.dtype))
