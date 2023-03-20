#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest
from numba import njit

from numba_dpex.tests._helper import (
    assert_auto_offloading,
    dpnp_debug,
    skip_no_dpnp,
    skip_no_opencl_gpu,
)


@skip_no_opencl_gpu
@skip_no_dpnp
class Testdpnp_functions:
    N = 10

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float32, np.double]

    def test_dpnp_interacting_with_parfor(self):
        def f(a, b):
            c = np.sum(a)
            e = np.add(b, a)
            d = c + e
            return d

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(
            device
        ), assert_auto_offloading(), dpnp_debug():
            njit_f = njit(f)
            got = njit_f(self.a, self.b)
        expected = f(self.a, self.b)

        max_abs_err = got.sum() - expected.sum()
        assert max_abs_err < 1e-4
