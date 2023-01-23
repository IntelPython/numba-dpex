# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpex


@ndpex.kernel
def atomic_reduction(a):
    idx = ndpex.get_global_id(0)
    ndpex.atomic.add(a, 0, a[idx])


def main():
    N = 10
    a = np.arange(N)

    print("Using device ...")
    print(a.device)

    atomic_reduction[N](a)
    print("Reduction sum =", a[0])

    print("Done...")


if __name__ == "__main__":
    main()
