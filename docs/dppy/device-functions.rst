Writing Device Functions
========================

DPPY device functions can only be invoked from within the device (by a kernel
or another device function). To define a device function::

    import numba_dppy as dppy

    @dppy.func
    def a_device_function(a):
        return a + 1

    @dppy.kernel
    def a_kernel_function(a, b):
        i = dppy.get_global_id(0)
        b[i] = a_device_function(a[i])

Unlike a kernel function, a device function can return a value like normal
functions.

Transition from Numba CUDA
--------------------------

Replace ``@cuda.jit(device=True)`` with ``@dppy.func``.

See also
--------

Examples:

- ``numba_dppy/examples/dppy_func.py``
