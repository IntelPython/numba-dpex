#! /usr/bin/env python

# Copyright 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as dpex
from numba_dpex.tests._helper import skip_no_opencl_cpu, skip_no_opencl_gpu


@dpex.kernel
def data_parallel_sum(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
d = a + b


@skip_no_opencl_cpu
class TestArrayArgsCPU:
    def test_device_array_args_cpu(self):
        c = np.ones_like(a)

        with dpctl.device_context("opencl:cpu"):
            data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

            assert np.all(c == d)


@skip_no_opencl_gpu
class TestArrayArgsGPU:
    def test_device_array_args_gpu(self):
        c = np.ones_like(a)

        with dpctl.device_context("opencl:gpu"):
            data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

        assert np.all(c == d)
