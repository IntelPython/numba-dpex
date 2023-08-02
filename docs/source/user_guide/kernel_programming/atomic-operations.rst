Supported Atomic Operations
===========================

Numba-dpex supports some of the atomic operations supported in DPC++.
Those that are presently implemented are as follows:

.. automodule:: numba_dpex.ocl.stubs
   :members: atomic
   :noindex:

Example
-------

Example usage of atomic operations

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/atomic_op.py
   :pyobject: main

.. note::

    The ``numba_dpex.atomic.add`` function is analogous to The
    ``numba.cuda.atomic.add`` provided by the ``numba.cuda`` backend.

Full examples
-------------

- :file:`numba_dpex/examples/atomic_op.py`
