# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpex

N = 10


@ndpex.kernel
def kernel_atomic_reduction(a):
    idx = ndpex.get_global_id(0)
    ndpex.atomic.add(a, 0, a[idx])


def main():
    try:
        a = np.arange(N, device="gpu")
    except:
        print("No GPU device")

    print("Using device ...")
    print(a.device)
    print("a=", a)

    kernel_atomic_reduction[N, ndpex.DEFAULT_LOCAL_SIZE](a)
    print("Reduction sum =", a[0])

    print("Done...")


if __name__ == "__main__":
    main()
