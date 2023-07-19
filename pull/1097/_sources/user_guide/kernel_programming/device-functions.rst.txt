Writing Device Functions
========================

The user-level API of SYCL does not have a notion for device-only functions,
*i.e.* functions that can be only invoked from a kernel and not from a host
function. However, numba-dpex provides a special decorator
``numba_dpex.func`` specifically to implement device functions.

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/device_func.py
   :pyobject: a_device_function

To use a device function from an another device function:

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/device_func.py
   :pyobject: another_device_function

To use a device function from a kernel function ``numba_dpex.kernel``:

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/device_func.py
   :pyobject: a_kernel_function

Unlike a kernel function, a device function can return a value like normal
functions.

.. todo::

   Specific capabilities and limitations for device functions need to be added.
