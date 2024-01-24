#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.memory as dpctl_mem
import dpctl.tensor as dpt
import numpy as np

from numba_dpex.utils import (
    as_usm_obj,
    copy_to_numpy_from_usm_obj,
    has_usm_memory,
)

from . import _helper


def test_has_usm_memory():
    a = np.ones(1023, dtype=np.float32)
    q = dpctl.SyclQueue()
    # test usm_ndarray
    da = dpt.usm_ndarray(a.shape, dtype=a.dtype, buffer="shared")
    usm_mem = has_usm_memory(da)
    assert da.usm_data._pointer == usm_mem._pointer

    # test usm allocated numpy.ndarray
    buf = dpctl_mem.MemoryUSMShared(a.size * a.dtype.itemsize, queue=q)
    ary_buf = np.ndarray(a.shape, buffer=buf, dtype=a.dtype)
    usm_mem = has_usm_memory(ary_buf)
    assert buf._pointer == usm_mem._pointer

    usm_mem = has_usm_memory(a)
    assert usm_mem is None


def test_as_usm_obj():
    a = np.ones(1023, dtype=np.float32)
    b = a * 3
    queue = dpctl.SyclQueue()
    a_copy = np.empty_like(a)
    usm_mem = as_usm_obj(a, queue=queue)
    copy_to_numpy_from_usm_obj(usm_mem, a_copy)
    assert np.all(a == a_copy)

    b_copy = np.empty_like(b)
    usm_mem = as_usm_obj(b, queue=queue, copy=False)
    copy_to_numpy_from_usm_obj(usm_mem, b_copy)
    assert np.any(np.not_equal(b, b_copy))
