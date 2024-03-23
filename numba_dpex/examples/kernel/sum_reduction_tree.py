# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

""" Demonstration of a simple tree reduction algorithm to reduce an array of scalars.

The algorithm works in two steps: First an nd-range kernel is launched to
calculate a partially reduced array. The size of the partially reduced array is
equal to the number of work groups over which the initial reduction is done.
The partial results are then summed up on the host device.
"""
import dpctl
import dpctl.tensor as dpt

import numba_dpex as ndpx
from numba_dpex import kernel_api as kapi


@ndpx.kernel
def sum_reduction_kernel(nditem: kapi.NdItem, A, partial_sums, slm):
    """
    The example demonstrates a reduction kernel implemented as a ``kernel``
    function.
    """
    local_id = nditem.get_local_id(0)
    global_id = nditem.get_global_id(0)
    group_size = nditem.get_local_range(0)
    gr = nditem.get_group()
    group_id = gr.get_group_id(0)

    # Copy from global to local memory
    slm[local_id] = A[global_id]

    # Loop for computing local_sums : divide workgroup into 2 parts
    stride = group_size // 2
    while stride > 0:
        # Waiting for each 2x2 addition into given workgroup
        kapi.group_barrier(gr)

        # Add elements 2 by 2 between local_id and local_id + stride
        if local_id < stride:
            slm[local_id] += slm[local_id + stride]

        stride >>= 1

    if local_id == 0:
        partial_sums[group_id] = slm[0]


def sum_reduce(A):
    global_size = len(A)
    work_group_size = 64
    # nb_work_groups have to be even for this implementation
    nb_work_groups = global_size // work_group_size

    partial_sums = dpt.zeros(nb_work_groups, dtype=A.dtype, device=A.device)

    gs = ndpx.Range(global_size)
    ls = ndpx.Range(work_group_size)
    slm = kapi.LocalAccessor(64, A.dtype)
    ndpx.call_kernel(
        sum_reduction_kernel, ndpx.NdRange(gs, ls), A, partial_sums, slm
    )

    final_sum = 0
    # calculate the final sum in HOST
    for i in range(nb_work_groups):
        final_sum += int(partial_sums[i])

    return final_sum


def test_sum_reduce():
    N = 1024
    device = dpctl.select_default_device()
    A = dpt.ones(N, dtype=dpt.int32, device=device)

    print("Running Device + Host reduction")

    actual = sum_reduce(A)
    expected = N

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert actual == expected

    print("Done...")


if __name__ == "__main__":
    test_sum_reduce()
