# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpx


@ndpx.kernel
def atomic_reduction(a, res):
    """Summarize all the items in a and writes it into res using atomic.add.

    :param a: array of values to get sum
    :param res: result where to add all the items from a array. It must be preset to 0.
    """
    idx = ndpx.get_global_id(0)
    ndpx.atomic.add(res, 0, a[idx])


def main():
    N = 10

    # We are storing sum to the first element
    a = np.arange(0, N)
    res = np.zeros(1, dtype=a.dtype)

    print("Using device ...")
    print(a.device)

    atomic_reduction[ndpx.Range(N)](a, res)
    print("Reduction sum =", res[0])

    print("Done...")


if __name__ == "__main__":
    main()
