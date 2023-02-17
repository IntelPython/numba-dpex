#! /usr/bin/env python

# Copyright 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest
from numba import njit, prange

from numba_dpex.tests._helper import assert_auto_offloading, skip_no_opencl_gpu


class TestPrange:
    def test_one_prange(self):
        @njit
        def f(a, b):
            for i in prange(4):
                b[i, 0] = a[i, 0] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        device = dpctl.select_default_device()
        with assert_auto_offloading(), dpctl.device_context(device):
            f(a, b)

        for i in range(4):
            assert b[i, 0] == a[i, 0] * 10

    def test_nested_prange(self):
        @njit
        def f(a, b):
            # dimensions must be provided as scalar
            m, n = a.shape
            for i in prange(m):
                for j in prange(n):
                    b[i, j] = a[i, j] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        device = dpctl.select_default_device()
        with assert_auto_offloading(), dpctl.device_context(device):
            f(a, b)

        assert np.all(b == 10)

    @pytest.mark.skip
    def test_multiple_prange(self):
        @njit
        def f(a, b):
            # dimensions must be provided as scalar
            m, n = a.shape
            for i in prange(m):
                val = 10
                for j in prange(n):
                    b[i, j] = a[i, j] * val

            for i in prange(m):
                for j in prange(n):
                    a[i, j] = a[i, j] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        device = dpctl.select_default_device()
        with assert_auto_offloading(parfor_offloaded=2), dpctl.device_context(
            device
        ):
            f(a, b)

        assert np.all(b == 10)
        assert np.all(a == 10)

    def test_three_prange(self):
        @njit
        def f(a, b):
            # dimensions must be provided as scalar
            m, n, o = a.shape
            for i in prange(m):
                val = 10
                for j in prange(n):
                    constant = 2
                    for k in prange(o):
                        b[i, j, k] = a[i, j, k] * (val + constant)

        m = 8
        n = 8
        o = 8
        a = np.ones((m, n, o))
        b = np.ones((m, n, o))

        device = dpctl.select_default_device()
        with assert_auto_offloading(parfor_offloaded=1), dpctl.device_context(
            device
        ):
            f(a, b)

        assert np.all(b == 12)

    @pytest.mark.skip(reason="numba-dpex issue 110")
    def test_two_consequent_prange(self):
        def prange_example():
            n = 10
            a = np.ones((n), dtype=np.float64)
            b = np.ones((n), dtype=np.float64)
            c = np.ones((n), dtype=np.float64)
            for i in prange(n // 2):
                a[i] = b[i] + c[i]

            return a

        jitted = njit(prange_example)

        device = dpctl.select_default_device()
        with assert_auto_offloading(parfor_offloaded=2), dpctl.device_context(
            device
        ):
            jitted_res = jitted()

        res = prange_example()

        np.testing.assert_equal(res, jitted_res)

    @pytest.mark.skip(reason="NRT required but not enabled")
    def test_2d_arrays(self):
        def prange_example():
            n = 10
            a = np.ones((n, n), dtype=np.float64)
            b = np.ones((n, n), dtype=np.float64)
            c = np.ones((n, n), dtype=np.float64)
            for i in prange(n // 2):
                a[i] = b[i] + c[i]

            return a

        jitted = njit(prange_example)

        device = dpctl.select_default_device()
        with assert_auto_offloading(parfor_offloaded=2), dpctl.device_context(
            device
        ):
            jitted_res = jitted()

        res = prange_example()

        np.testing.assert_equal(res, jitted_res)
