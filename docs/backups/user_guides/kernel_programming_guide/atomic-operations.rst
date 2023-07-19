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

.. literalinclude:: ../../../numba_dpex/examples/atomic_op.py
   :pyobject: main

.. note::

    The ``numba_dpex.atomic.add`` function is analogous to The
    ``numba.cuda.atomic.add`` provided by the ``numba.cuda`` backend.

Generating Native FP Atomics
----------------------------
Numba-dpex supports generating native floating-point atomics.
This feature is experimental. Users will need to provide
the following environment variables to activate it.

    NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1

Example command:

.. code-block:: bash

    NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1 \
    python program.py

Full examples
-------------

- ``numba_dpex/examples/atomic_op.py``
