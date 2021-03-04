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
from numba import njit
import dpctl
from numba_dppy.context import device_context
import unittest
from numba_dppy.testing import ensure_dpnp


@unittest.skipUnless(
    ensure_dpnp() and dpctl.has_gpu_queues(), "test only when dpNP and GPU is available"
)
class Testdpnp_functions(unittest.TestCase):
    N = 10

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]

    def test_dpnp_interacting_with_parfor(self):
        def f(a, b):
            c = np.sum(a)
            e = np.add(b, a)
            d = c + e
            return d

        njit_f = njit(f)
        got = njit_f(self.a, self.b)
        expected = f(self.a, self.b)

        max_abs_err = got.sum() - expected.sum()
        self.assertTrue(max_abs_err < 1e-4)


if __name__ == "__main__":
    unittest.main()
