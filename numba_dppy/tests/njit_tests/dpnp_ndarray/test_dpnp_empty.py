"""Tests for DPNP ndarray constructors."""

import dpnp
from numba import njit


def test_dpnp_empty():
    @njit
    def func():
        dpnp.empty(10)

    func()
