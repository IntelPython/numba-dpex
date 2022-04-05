"""Tests for DPNP ndarray constructors."""

import dpnp
import pytest
from numba import njit

shapes = [10, (2, 5)]
dtypes = ["f8", dpnp.float32]
usm_types = ["device", "shared", "host"]


@pytest.mark.skip(reason="functional is not implemented")
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_ones(shape, dtype, usm_type):
    from numba_dppy.dpctl_iface import get_current_queue

    @njit
    def func(shape):
        queue = get_current_queue()
        dpnp.ones(shape, dtype, usm_type, queue)

    func(shape)
