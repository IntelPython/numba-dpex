# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dpnp ndarray constructors."""

import dpnp
import pytest
from numba import njit

shapes = [10, (2, 5)]
dtypes = ["f8", dpnp.float32]
usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_empty(shape, dtype, usm_type):
    from numba_dpex.dpctl_iface import get_current_queue

    @njit
    def func(shape):
        queue = get_current_queue()
        dpnp.empty(shape, dtype, usm_type, queue)

    func(shape)
