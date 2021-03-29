Synchronization Functions
=========================

Currently, ``numba-dppy`` only supports some of the SYCL synchronization operations. For synchronization of all threads in the same thread block, ``numba-dppy`` provides a helper function called ``numba_dppy.barrier()``. This function implements the same pattern as barriers in traditional
multi-threaded programming: invoking the function forces a thread to wait until
all threads in the block reach the barrier, at which point it returns control
to all its callers.

``numba_dppy.barrier()`` supports two memory fence options:

- ``numba_dppy.CLK_GLOBAL_MEM_FENCE``: The barrier function will queue a memory
  fence to ensure correct ordering of memory operations to global memory. Using
  the option can be useful when work-items, for example, write to buffer or
  image objects and then want to read the updated data. Passing no arguments to ``numba_dppy.barrier()`` is equivalent to setting the global memory fence option. For example,

  .. literalinclude:: ../../../numba_dppy/examples/barrier.py
   :pyobject: no_arg_barrier_support

- ``numba_dppy.CLK_LOCAL_MEM_FENCE``: The barrier function will either flush
  any variables stored in local memory or queue a memory fence to ensure
  correct ordering of memory operations to local memory. For example,

.. literalinclude:: ../../../numba_dppy/examples/barrier.py
   :pyobject: local_memory


.. note::

    The ``numba_dppy.barrier()`` function is semantically equivalent to
    ``numba.cuda.syncthreads``.
