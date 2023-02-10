# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpx


@ndpx.kernel
def atomic_reduction(a):
    idx = ndpx.get_global_id(0)
    ndpx.atomic.add(a, 0, a[idx])


def main():
    N = 10
    a = np.arange(N)

    print("Using device ...")
    print(a.device)

    atomic_reduction[ndpx.Range(N)](a)
    print("Reduction sum =", a[0])

    print("Done...")


if __name__ == "__main__":
    main()
