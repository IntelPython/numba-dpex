#! /usr/bin/env python
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dpctl
import dpctl.memory as dpctl_mem
import dpctl.tensor as dpt
import numpy as np
import pytest

from numba_dppy.tests._helper import skip_test
from numba_dppy.utils import (
    as_usm_obj,
    copy_to_numpy_from_usm_obj,
    has_usm_memory,
)

from . import _helper


def test_has_usm_memory(offload_device):
    if skip_test(offload_device):
        pytest.skip()

    a = np.ones(1023, dtype=np.float32)

    with dpctl.device_context(offload_device):
        # test usm_ndarray
        da = dpt.usm_ndarray(a.shape, dtype=a.dtype, buffer="shared")
        usm_mem = has_usm_memory(da)
        assert da.usm_data._pointer == usm_mem._pointer

        # test usm allocated numpy.ndarray
        buf = dpctl_mem.MemoryUSMShared(a.size * a.dtype.itemsize)
        ary_buf = np.ndarray(a.shape, buffer=buf, dtype=a.dtype)
        usm_mem = has_usm_memory(ary_buf)
        assert buf._pointer == usm_mem._pointer

        usm_mem = has_usm_memory(a)
        assert usm_mem is None


def test_as_usm_obj(offload_device):
    if skip_test(offload_device):
        pytest.skip()

    a = np.ones(1023, dtype=np.float32)
    b = a * 3

    with dpctl.device_context(offload_device) as queue:
        a_copy = np.empty_like(a)
        usm_mem = as_usm_obj(a, queue=queue)
        copy_to_numpy_from_usm_obj(usm_mem, a_copy)
        assert np.all(a == a_copy)

        b_copy = np.empty_like(b)
        usm_mem = as_usm_obj(b, queue=queue, copy=False)
        copy_to_numpy_from_usm_obj(usm_mem, b_copy)
        assert np.any(np.not_equal(b, b_copy))
