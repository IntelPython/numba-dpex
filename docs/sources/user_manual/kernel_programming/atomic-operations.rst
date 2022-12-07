Atomic Operations
=================

Atomic operations are the operations with local or global memory that ensure no race condition can happen
if several parallel threads access this memory.

**Data Parallel Extension for Numba** supports a few essential atomic operations:

.. list-table::
   :widths: 150 600
   :header-rows: 1

   * - Atomic function
     - Description
   * - ``numba_dpex.atomic.add(ary, idx, val)``
     - Performs atomic addition ``ary[idx] += val``.

        Parameters:
           ``ary``: An array on which the atomic operation is performed.
                Allowed types: ``int32``, ``int64``, ``float32``, or ``float64``

           ``idx`` (``int``): Index of the array element, on which atomic operation is performed

           ``val``: The value of an increment.
                Its type must match the type of array elements, ``ary[]``

        Returns:
               The old value at the index location ``ary[idx]`` as if it is loaded atomically.
   * - ``numba_dpex.atomic.sub(ary, idx, val)``
     - Performs atomic subtraction ``ary[idx] -= val``.

        Parameters:
           ``ary``: An array on which the atomic operation is performed.
                Allowed types: ``int32``, ``int64``, ``float32``, or ``float64``

           ``idx`` (``int``): Index of the array element, on which atomic operation is performed

           ``val``: The value of a decrement.
                Its type must match the type of array elements, ``ary[]``

        Returns:
               The old value at the index location ``ary[idx]`` as if it is loaded atomically.

Example usage of atomic operations:

.. literalinclude:: ../../../../numba_dpex/examples/kernel/atomic_reduction.py
    :lines: 5-

.. note::

    The ``numba_dpex.atomic.add`` and ``numba_dpex.atomic.sub`` functions are analogous to
    ``numba.cuda.atomic.add`` and ``numba.cuda.atomic.sub`` respectively.
