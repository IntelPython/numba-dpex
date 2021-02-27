.. _device-functions:

Writing Device Functions
========================

OpenCL and SYCL do not directly have a notion for device-only functions, *i.e.*
functions that can be only invoked from a kernel and not a host function.
However, ``numba-dppy`` provides a special decorator ``numba_dppy.func``
specifically to implement device functions.

.. literalinclude:: ../../../numba_dppy/examples/dppy_func.py
   :pyobject: g

Device functions can only be invoked from inside kernel functions ``numba_dppy.kernel``.

.. literalinclude:: ../../../numba_dppy/examples/dppy_func.py
   :pyobject: f

Unlike a kernel function, a device function can return a value like normal
functions.

.. todo::

   Specific capabilities and limitations for device functions need to be added.
