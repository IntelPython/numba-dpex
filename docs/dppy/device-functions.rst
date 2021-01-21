Writing Device Functions
========================

DPPY device functions can only be invoked from within the device (by a kernel
or another device function). To define a device function:

.. literalinclude:: ../../numba_dppy/examples/dppy_func.py
   :pyobject: a_device_function

To use a device function from an another device function:

.. literalinclude:: ../../numba_dppy/examples/dppy_func.py
   :pyobject: an_another_device_function

To use a device function from a kernel:

.. literalinclude:: ../../numba_dppy/examples/dppy_func.py
   :pyobject: a_kernel_function

Unlike a kernel function, a device function can return a value like normal
functions.

Transition from Numba CUDA
--------------------------

Replace ``@cuda.jit(device=True)`` with ``@dppy.func``.

See also
--------

Examples:

- ``numba_dppy/examples/dppy_func.py``
