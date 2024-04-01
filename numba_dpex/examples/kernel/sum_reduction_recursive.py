# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
There are multiple ways of implementing reduction using numba_ndpx. Here we
demonstrate another way of implementing reduction using recursion to compute
partial reductions in separate kernels.
"""

import dpctl
import dpctl.tensor as dpt
from numba import int32

import numba_dpex as ndpx
from numba_dpex import kernel_api as kapi


@ndpx.kernel
def sum_reduction_kernel(nditem: kapi.NdItem, A, input_size, partial_sums, slm):
    local_id = nditem.get_local_id(0)
    global_id = nditem.get_global_id(0)
    group_size = nditem.get_local_range(0)
    gr = nditem.get_group()
    group_id = gr.get_group_id(0)
    slm[local_id] = 0

    if global_id < input_size:
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


def sum_recursive_reduction(size, group_size, Dinp, Dpartial_sums):
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

    gr = ndpx.Range(passed_size)
    lr = ndpx.Range(group_size)
    slm = kapi.LocalAccessor(64, Dinp.dtype)
    ndpx.call_kernel(
        sum_reduction_kernel,
        ndpx.NdRange(gr, lr),
        Dinp,
        size,
        Dpartial_sums,
        slm,
    )

    if nb_work_groups <= group_size:
        ndpx.call_kernel(
            sum_reduction_kernel,
            ndpx.NdRange(lr, lr),
            Dpartial_sums,
            nb_work_groups,
            Dinp,
            slm,
        )
        result = int(Dinp[0])
    else:
        result = sum_recursive_reduction(
            nb_work_groups, group_size, Dpartial_sums, Dinp
        )

    return result


def sum_reduce(A):
    global_size = len(A)
    work_group_size = 64
    nb_work_groups = global_size // work_group_size
    if (global_size % work_group_size) != 0:
        nb_work_groups += 1

    partial_sums = dpt.zeros(nb_work_groups, dtype=A.dtype, device=A.device)
    result = sum_recursive_reduction(
        global_size, work_group_size, A, partial_sums
    )

    return result


def test_sum_reduce():
    N = 20000
    device = dpctl.select_default_device()
    A = dpt.ones(N, dtype=dpt.int32, device=device)

    print("Running recursive reduction")

    actual = sum_reduce(A)
    expected = N

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert actual == expected

    print("Done...")


if __name__ == "__main__":
    test_sum_reduce()
