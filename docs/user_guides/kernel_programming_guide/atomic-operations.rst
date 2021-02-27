Supported Atomic Operations
===========================

``numba-dppy`` supports some of the atomic operations supported in DPC++.
Those that are presently implemented are as follows:

.. automodule:: numba_dppy.ocl.stubs
   :members: atomic
   :noindex:

Example
-------

Here's an example of how to use atomics add in DPPY:

.. literalinclude:: ../../../numba_dppy/examples/atomic_op.py
   :pyobject: main

.. note::

    The ``numba_dppy.atomic.add`` function is analogous to The
    ``numba.cuda.atomic.add`` provided by the ``numba.cuda`` backend.

Full examples
-------------

- ``numba_dppy/examples/atomic_op.py``
