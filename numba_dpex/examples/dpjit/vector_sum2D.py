# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

from numba_dpex import dpjit


@dpjit
def f1(a, b):
    c = a + b
    return c


N = 1000
print("N", N)

a = np.ones((N, N), dtype=np.float32)
b = np.ones((N, N), dtype=np.float32)

print("a:", a)
print("b:", b)


def main():
    c = f1(a, b)
    print("c:", c)


if __name__ == "__main__":
    main()
