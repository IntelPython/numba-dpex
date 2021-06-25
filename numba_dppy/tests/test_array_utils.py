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

import numpy as np
import pytest
import dpctl
import dpctl.tensor as dpt
import dpctl.memory as dpctl_mem
from . import _helper
from numba_dppy.utils import (is_usm_backed,
                              as_usm_backed,
                              copy_from_usm_backed)


def test_is_usm_backed(offload_device):
    a = np.ones(1023, dtype=np.float32)

    with dpctl.device_context(offload_device):
        # test usm_ndarray
        da = dpt.usm_ndarray(a.shape, dtype=a.dtype, buffer="shared")
        usm_mem = is_usm_backed(da)
        assert da.usm_data._pointer == usm_mem._pointer

        # test usm backed numpy.ndarray
        buf = dpctl_mem.MemoryUSMShared(a.size * a.dtype.itemsize)
        ary_buf = np.ndarray(a.shape, buffer=buf, dtype=a.dtype)
        usm_mem = is_usm_backed(ary_buf)
        assert buf._pointer == usm_mem._pointer

        usm_mem = is_usm_backed(a)
        assert usm_mem is None


def test_as_usm_backed(offload_device):
    a = np.ones(1023, dtype=np.float32)

    with dpctl.device_context(offload_device) as queue:
        a_copy = np.empty_like(a)
        usm_mem = as_usm_backed(a, queue=queue)
        copy_from_usm_backed(usm_mem, a_copy)
        assert np.all(a == a_copy)

        a_copy = np.empty_like(a)
        usm_mem = as_usm_backed(a, queue=queue, copy=False)
        copy_from_usm_backed(usm_mem, a_copy)
        assert np.any(np.not_equal(a, a_copy))
