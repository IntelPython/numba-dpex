Atomic Operations
=================

Atomic operations are the operations with local or global memory that ensure no race condition can happen
if several parallel threads access this memory.

**Data Parallel Extension for Numba** supports a few essential atomic operations:

.. automodule:: numba_dpex.ocl.stubs.atomic
   :members:
   :noindex:

Example usage of atomic operations

.. literalinclude:: ../../../numba_dpex/examples/atomic_op.py
   :pyobject: main

.. note::

    The ``numba_dpex.atomic.add`` function is analogous to The
    ``numba.cuda.atomic.add``.

Generating Native FP Atomics
----------------------------
Numba-dpex supports generating native floating-point atomics.
This feature is experimental. Users will need to provide
the following environment variables to activate it.

    NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1
    NUMBA_DPEX_LLVM_SPIRV_ROOT=/path/to/dpcpp/provided/llvm_spirv

Example command:

.. code-block:: bash

    NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1 \
    NUMBA_DPEX_LLVM_SPIRV_ROOT=/path/to/dpcpp/provided/llvm_spirv \
    python program.py

Full examples
-------------

- ``numba_dpex/examples/atomic_op.py``
