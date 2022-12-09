################################################################################
#                                 Numba-DPEX
#
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
################################################################################

import dpctl.tensor.numpy_usm_shared
import dpnp
import numba
import pytest

containers = [
    dpctl.tensor.numpy_usm_shared.ndarray([10]),
    dpnp.ndarray([10]),
]

pipelines = [numba.njit]


@pytest.mark.parametrize("container", containers)
@pytest.mark.parametrize("pipeline", pipelines)
def test_unboxing(container, pipeline):
    @pipeline
    def func(a):
        return 0

    func(container)


@pytest.mark.parametrize("container", containers)
@pytest.mark.parametrize("pipeline", pipelines)
def test_boxing(container, pipeline):
    @pipeline
    def func(a):
        return a

    func(container)
