Supported Atomic Operations
===========================

Numba provides access to some of the atomic operations supported in DPPY.
Those that are presently implemented are as follows:

.. automodule:: numba_dppy.ocl.stubs
   :members: atomic
   :noindex:

Example
-------

Here's an example of how to use atomics add in DPPY:

.. literalinclude:: ../../numba_dppy/examples/atomic_op.py
   :pyobject: main

Transition from Numba CUDA
--------------------------

Replace ``numba.cuda.atomic.add`` with ``dppy.atomic.add``.

See also
--------

Examples:

- ``numba_dppy/examples/atomic_op.py``
