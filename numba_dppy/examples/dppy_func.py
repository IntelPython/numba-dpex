import numpy as np
import numba_dppy as dppy
import dpctl


@dppy.func
def a_device_function(a):
    return a + 1


@dppy.kernel
def a_kernel_function(a, b):
    i = dppy.get_global_id(0)
    b[i] = a_device_function(a[i])


def driver(a, b, N):
    print(b)
    print("--------")
    a_kernel_function[N, dppy.DEFAULT_LOCAL_SIZE](a, b)
    print(b)


def get_context():
    if dpctl.has_gpu_queues():
        return "opencl:gpu"
    elif dpctl.has_cpu_queues():
        return "opencl:cpu"
    else:
        raise RuntimeError("No device found")

def main():
    N = 10
    a = np.ones(N)
    b = np.ones(N)

    context = get_context()

    print("Device Context:", context)

    with dpctl.device_context(context):
        driver(a, b, N)


if __name__ == "__main__":
    main()
