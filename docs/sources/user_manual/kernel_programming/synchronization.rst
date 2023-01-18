Synchronization Functions
=========================

**Data Parallel Extension for Numba** supports some synchronization operations. For
synchronization of all threads in the same thread block, ``numba-dpex`` provides
a helper function called ``numba_dpex.barrier()``. This function implements the
same pattern as barriers in traditional multi-threaded programming: invoking the
function forces a thread to wait until all threads in the block reach the
barrier, at which point it returns control to all its callers.

``numba_dpex.barrier(fence_type)`` supports two memory fence options.

.. list-table:: **Global and local memory fences**
   :widths: 30 100
   :header-rows: 1

   * - ``fence_type``
     - Description
   * - ``numba_dpex.CLK_GLOBAL_MEM_FENCE``
     - The barrier function will queue a memory
       fence to ensure correct ordering of memory operations to global memory. Using
       the option can be useful when updating all work-items and then reading the updated data.
       Passing no arguments to ``numba_dpex.barrier()`` is equivalent to setting the global memory fence.
   * - ``numba_dpex.CLK_LOCAL_MEM_FENCE``
     - The barrier function will either flush any variables stored in local memory or
       queue a memory fence to ensure correct ordering of memory operations to local memory.


The following example illustrates the difference between local and global memory fences.

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/scan.py
      :lines: 5-
      :emphasize-lines: 23, 33, 43
      :linenos:

In the line 22 the global data ``a[gid]`` is being copied into local arrays ``c[lid]`` and ``b[lid]``.
The local barrier in line 23 ensures no usage of ``b[]`` and ``c[]`` happens in the work group until this copy
is complete.

Likewise, in the line 33 the local barrier ensures the local data computed in lines 28-31 is up to date
before proceeding to its usage in lines 36-38.

Finally, the global memory fence is used in the line 43 to ensure it is up to date before writing into a global
memory in the line 44.

.. note::

    The ``numba_dpex.barrier()`` function is semantically equivalent to
    ``numba.cuda.syncthreads``.
