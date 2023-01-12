#! /usr/bin/env python

# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt

import numba_dpex as dpex
from numba_dpex.tests._helper import skip_no_opencl_cpu, skip_no_opencl_gpu


@dpex.kernel
def data_parallel_sum(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
N = global_size

a = dpt.ones(N, dtype=dpt.float32)
b = dpt.ones(N, dtype=dpt.float32)


@skip_no_opencl_cpu
class TestArrayArgsGPU:
    def test_device_array_args_cpu(self):
        c = dpt.ones_like(a)

        with dpctl.device_context("opencl:cpu"):
            data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

        npc = dpt.asnumpy(c)
        import numpy as np

        npc_expected = np.full(N, 2.0, dtype=np.float32)
        assert np.all(npc == npc_expected)


@skip_no_opencl_gpu
class TestArrayArgsCPU:
    def test_device_array_args_gpu(self):
        c = dpt.ones_like(a)

        with dpctl.device_context("opencl:gpu"):
            data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

        npc = dpt.asnumpy(c)
        import numpy as np

        npc_expected = np.full(N, 2.0, dtype=np.float32)
        assert np.all(npc == npc_expected)
