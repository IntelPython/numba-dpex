from numba import njit
import numpy as np
import dpctl


@njit
def f1(a, b):
    c = a + b
    return c


def main():
    global_size = 64
    local_size = 32
    N = global_size * local_size
    print("N", N)

    a = np.ones(N, dtype=np.float32)
    b = np.ones(N, dtype=np.float32)

    print("a:", a, hex(a.ctypes.data))
    print("b:", b, hex(b.ctypes.data))

    with dpctl.device_context("opencl:gpu:0"):
        c = f1(a, b)

    print("RESULT c:", c, hex(c.ctypes.data))
    for i in range(N):
        if c[i] != 2.0:
            print("First index not equal to 2.0 was", i)
            break


if __name__ == '__main__':
    main()
