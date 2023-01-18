# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import numpy as np
from numba import float32

import numba_dpex


def private_memory():
    """
    This example demonstrates the usage of numba_dpex's `private.array`
    intrinsic function. The function is used to create a static array
    allocated on the devices private address space.
    """

    @numba_dpex.kernel
    def private_memory_kernel(A):
        memory = numba_dpex.private.array(shape=1, dtype=np.float32)
        i = numba_dpex.get_global_id(0)

        # preload
        memory[0] = i
        numba_dpex.barrier(numba_dpex.LOCAL_MEM_FENCE)  # local mem fence

        # memory will not hold correct deterministic result if it is not
        # private to each thread.
        A[i] = memory[0] * 2

    N = 4
    device = dpctl.select_default_device()

    arr = dpt.zeros(N, dtype=dpt.float32, device=device)
    orig = np.arange(N).astype(np.float32)

    print("Using device ...")
    device.print_device_info()

    global_range = (N,)
    local_range = (N,)
    private_memory_kernel[global_range, local_range](arr)

    arr_out = dpt.asnumpy(arr)
    np.testing.assert_allclose(orig * 2, arr_out)
    # the output should be `orig[i] * 2, i.e. [0, 2, 4, ..]``
    print(arr_out)


def main():
    private_memory()

    print("Done...")


if __name__ == "__main__":
    main()
