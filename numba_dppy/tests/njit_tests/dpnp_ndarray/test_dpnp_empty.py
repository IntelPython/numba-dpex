"""Tests for DPNP ndarray constructors."""

import dpnp
from numba import njit


def test_dpnp_empty():
    from numba_dppy.dpctl_iface import get_current_queue

    @njit
    def func():
        queue = get_current_queue()
        dpnp.empty(10, usm_type="device", sycl_queue=queue)

    func()
