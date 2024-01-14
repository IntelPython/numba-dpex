# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for boxing for dpctl.SyclQueue
"""

import dpnp
import pytest
from dpctl import SyclQueue

from numba_dpex import dpjit


def test_boxing_without_parent():
    """Test unboxing of the queue that does not have parent"""

    @dpjit
    def func() -> SyclQueue:
        arr = dpnp.empty(10)
        queue = arr.sycl_queue
        return queue

    q: SyclQueue = func()

    assert len(q.sycl_device.filter_string) > 0
