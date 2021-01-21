import numpy as np
from numba import int32
import numba_dppy as dppy
import math

import dpctl


@dppy.kernel
def reduction_kernel(inp, partial_sums):
    local_id = dppy.get_local_id(0)
    global_id = dppy.get_global_id(0)
    group_size = dppy.get_local_size(0)
    group_id = dppy.get_group_id(0)

    local_sums = dppy.local.static_alloc(64, int32)

    # Copy from global to local memory
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


def get_context():
    if dpctl.has_gpu_queues():
        return "opencl:gpu"
    elif dpctl.has_cpu_queues():
        return "opencl:cpu"
    else:
        raise RuntimeError("No device found")


def sum_reduce(inp):
    global_size = len(inp)
    work_group_size = 64
    # nb_work_groups have to be even for this implementation
    nb_work_groups = global_size // work_group_size

    partial_sums = np.zeros(nb_work_groups).astype(np.int32)

    context = get_context()
    with dpctl.device_context(context):
        reduction_kernel[global_size, work_group_size](inp, partial_sums)

    final_sum = 0
    # calculate the final sum in HOST
    for i in range(nb_work_groups):
        final_sum += partial_sums[i]

    return final_sum

def test_sum_reduce():
    N = 1024
    inp = np.ones(N).astype(np.int32)

    print("Running Device + Host reduction")

    actual = sum_reduce(inp)
    expected = N

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert actual == expected


if __name__ == "__main__":
    test_sum_reduce()
