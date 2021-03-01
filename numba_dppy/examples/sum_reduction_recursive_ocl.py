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

import sys
import numpy as np
from numba import int32
import numba_dppy, numba_dppy as dppy
import math

import dpctl
import dpctl.memory as dpctl_mem


def recursive_reduction(size, group_size, Dinp, Dpartial_sums):
    @dppy.kernel
    def sum_reduction_kernel(inp, input_size, partial_sums):
        local_id = dppy.get_local_id(0)
        global_id = dppy.get_global_id(0)
        group_size = dppy.get_local_size(0)
        group_id = dppy.get_group_id(0)

        local_sums = dppy.local.array(64, int32)

        local_sums[local_id] = 0

        if global_id < input_size:
            local_sums[local_id] = inp[global_id]

        # Loop for computing local_sums : divide workgroup into 2 parts
        stride = group_size // 2
        while stride > 0:
            # Waiting for each 2x2 addition into given workgroup
            dppy.barrier(dppy.CLK_LOCAL_MEM_FENCE)

            # Add elements 2 by 2 between local_id and local_id + stride
            if local_id < stride:
                local_sums[local_id] += local_sums[local_id + stride]

            stride >>= 1

        if local_id == 0:
            partial_sums[group_id] = local_sums[0]

    result = 0
    nb_work_groups = 0
    passed_size = size

    if size <= group_size:
        nb_work_groups = 1
    else:
        nb_work_groups = size // group_size
        if size % group_size != 0:
            nb_work_groups += 1
            passed_size = nb_work_groups * group_size

    sum_reduction_kernel[passed_size, group_size](Dinp, size, Dpartial_sums)

    if nb_work_groups <= group_size:
        sum_reduction_kernel[group_size, group_size](
            Dpartial_sums, nb_work_groups, Dinp
        )
        result = Dinp[0]
    else:
        result = recursive_reduction(nb_work_groups, group_size, Dpartial_sums, Dinp)

    return result


def sum_reduction_recursive():
    global_size = 20000
    work_group_size = 64
    nb_work_groups = global_size // work_group_size
    if (global_size % work_group_size) != 0:
        nb_work_groups += 1

    inp = np.ones(global_size).astype(np.int32)
    partial_sums = np.zeros(nb_work_groups).astype(np.int32)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            inp_buf = dpctl_mem.MemoryUSMShared(inp.size * inp.dtype.itemsize)
            inp_ndarray = np.ndarray(inp.shape, buffer=inp_buf, dtype=inp.dtype)
            np.copyto(inp_ndarray, inp)

            partial_sums_buf = dpctl_mem.MemoryUSMShared(
                partial_sums.size * partial_sums.dtype.itemsize
            )
            partial_sums_ndarray = np.ndarray(
                partial_sums.shape, buffer=partial_sums_buf, dtype=partial_sums.dtype
            )
            np.copyto(partial_sums_ndarray, partial_sums)

            print("Running recursive reduction")
            result = recursive_reduction(
                global_size, work_group_size, inp_ndarray, partial_sums_ndarray
            )
    else:
        print("No device found")
        exit()

    print("Expected:", global_size, "--- GOT:", result)
    assert result == global_size


sum_reduction_recursive()
