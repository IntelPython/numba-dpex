"""Tests for DPNP ndarray constructors."""

import dpnp
import pytest
from numba import njit


@pytest.mark.parametrize("shape", [10, (2, 5)])
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_dpnp_empty(shape, usm_type):
    from numba_dppy.dpctl_iface import get_current_queue

    @njit
    def func(shape):
        queue = get_current_queue()
        dpnp.empty(shape, usm_type=usm_type, sycl_queue=queue)

    func(shape)
