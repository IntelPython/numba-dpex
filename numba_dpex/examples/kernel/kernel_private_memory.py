# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
""" This example demonstrates the usage of :class:`numba_dpex.kernel_api.PrivateArray`.

A ``PrivateArray`` is an array allocated on a device's private address space. As
the amount of available private memory is usually limited, programmers should be
careful with the size of a ``PrivateArray``. Allocating an array that is too big
will cause the data to be spilled to global memory, causing adverse performance.
"""

import dpctl
import dpctl.tensor as dpt
import numpy as np

import numba_dpex as ndpx
from numba_dpex import float32
from numba_dpex import kernel_api as kapi


def private_memory():
    """Demonstrates usage of :class:`numba_dpex.kernel_api.PrivateArray`."""

    @ndpx.kernel
    def private_memory_kernel(nditem: kapi.NdItem, A):
        memory = kapi.PrivateArray(shape=1, dtype=np.float32)
        i = nditem.get_global_id(0)

        # preload
        memory[0] = i
        gr = nditem.get_group()
        # local mem fence
        kapi.group_barrier(gr)

        # memory will not hold correct deterministic result if it is not
        # private to each thread.
        A[i] = memory[0] * 2

    N = 100
    device = dpctl.select_default_device()

    arr = dpt.zeros(N, dtype=dpt.float32, device=device)
    orig = np.arange(N).astype(np.float32)

    print("Executing on device:")
    device.print_device_info()

    global_range = ndpx.Range(N)
    local_range = ndpx.Range(N)
    ndpx.call_kernel(
        private_memory_kernel, ndpx.NdRange(global_range, local_range), arr
    )

    arr_out = dpt.asnumpy(arr)
    np.testing.assert_allclose(orig * 2, arr_out)
    # the output should be `orig[i] * 2, i.e. [0, 2, 4, ..]``
    print(arr_out)


def main():
    private_memory()

    print("Done...")


if __name__ == "__main__":
    main()
