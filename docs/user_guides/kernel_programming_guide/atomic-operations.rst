Supported Atomic Operations
===========================

Numba-dppy supports some of the atomic operations supported in DPC++.
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

Generating Native FP Atomics
----------------------------
Numba-dppy supports generating native floating-point atomics.
This feature is experimental. Users will need to provide
the following environment variables to activate it.

    NUMBA_DPPY_ACTIVATE_ATOMICS_FP_NATIVE=1
    NUMBA_DPPY_LLVM_SPIRV_ROOT=/path/to/dpcpp/provided/llvm_spirv

Example command:

.. code-block:: bash

    NUMBA_DPPY_ACTIVATE_ATOMICS_FP_NATIVE=1 \
    NUMBA_DPPY_LLVM_SPIRV_ROOT=/path/to/dpcpp/provided/llvm_spirv \
    python program.py

Full examples
-------------

- ``numba_dppy/examples/atomic_op.py``
