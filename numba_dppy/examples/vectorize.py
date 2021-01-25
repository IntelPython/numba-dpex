import numpy as np
from numba import vectorize
import dpctl


@vectorize(nopython=True)
def ufunc_kernel(x, y):
    return x + y


def get_context():
    if dpctl.has_gpu_queues():
        return "opencl:gpu"
    elif dpctl.has_cpu_queues():
        return "opencl:cpu"
    else:
        raise RuntimeError("No device found")


def test_ufunc():
    N = 10
    dtype = np.float64

    A = np.arange(N, dtype=dtype)
    B = np.arange(N, dtype=dtype) * 10

    context = get_context()
    with dpctl.device_context(context):
        C = ufunc_kernel(A, B)

    print(C)


if __name__ == "__main__":
    test_ufunc()
