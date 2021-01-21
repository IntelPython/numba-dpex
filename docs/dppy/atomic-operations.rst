Supported Atomic Operations
===========================

Numba provides access to some of the atomic operations supported in DPPY. Those that are presently implemented are as follows:

.. function:: class add(ary, idx, val)
Perform atomic ary[idx] += val. Returns the old value at the index location as if it is loaded atomically.

.. note:: Supported on int32, int64, float32, float64 operands only.

.. function:: class sub(ary, idx, val)
Perform atomic ary[idx] -= val. Returns the old value at the index location as if it is loaded atomically.

.. note:: Supported on int32, int64, float32, float64 operands only.

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