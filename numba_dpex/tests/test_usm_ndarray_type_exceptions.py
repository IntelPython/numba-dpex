# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""This is to test USMNdArray raising correct exceptions."""

import dpnp
import pytest
from numba import njit
from numba.core.errors import TypingError

from numba_dpex.dpctl_iface import get_current_queue

arguments = [
    ("shape=10", 'device="cpu"', "queue=a.sycl_queue"),
    ("shape=10", "device=10", "queue=a.sycl_queue"),
    ("shape=10", 'device="cpu"', "queue=test"),
    ("shape=10", 'device="dpu"'),
]


@pytest.mark.parametrize("argument", arguments)
def test_usm_ndarray_type_exceptions(argument):
    a = dpnp.ndarray(10)

    @njit
    def func(a):
        dpnp.empty(argument)

    with pytest.raises(TypingError):
        func(a)
