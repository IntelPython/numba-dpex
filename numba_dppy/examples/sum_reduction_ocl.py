# Copyright 2020, 2021 Intel Corporation
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
from numba import int32
import numba_dppy as dppy
import numpy as np


@dppy.kernel
def sum_reduction_kernel(A, partial_sums):
    """
    The example demonstrates a reduction kernel implemented as a ``kernel``
    function.
    """
    local_id = dppy.get_local_id(0)
    global_id = dppy.get_global_id(0)
    group_size = dppy.get_local_size(0)
    group_id = dppy.get_group_id(0)

    local_sums = dppy.local.array(64, int32)

    # Copy from global to local memory
    local_sums[local_id] = A[global_id]

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


def sum_reduce(A):
    global_size = len(A)
    work_group_size = 64
    # nb_work_groups have to be even for this implementation
    nb_work_groups = global_size // work_group_size

    partial_sums = np.zeros(nb_work_groups).astype(A.dtype)

    try:
        device = dpctl.select_default_device()
        with dpctl.device_context(device):
            print("Using device ...")
            device.print_device_info()
            sum_reduction_kernel[global_size, work_group_size](A, partial_sums)
        final_sum = 0
        # calculate the final sum in HOST
        for i in range(nb_work_groups):
            final_sum += partial_sums[i]

        return final_sum
    except ValueError:
        print("No SYCL GPU device found. Failed to perform the summation.")
        return -1


def test_sum_reduce():
    N = 1024
    A = np.ones(N).astype(np.int32)

    print("Running Device + Host reduction")

    actual = sum_reduce(A)
    expected = N

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert actual == expected


if __name__ == "__main__":
    test_sum_reduce()
