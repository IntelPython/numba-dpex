For synchronization of all threads in the same thread block, dppy provides
``dppy.barrier()``. This function implements the same pattern as barriers in traditional
multi-threaded programming: this function waits until all threads in the block
call it, at which point it returns control to all its callers.


To go from CUDA to DPPY, replace ``numba.cuda.syncthreads`` with
``dppy.barrier()``.

This function supports two type of memory fences.

- dppy.CLK_GLOBAL_MEM_FENCE: The barrier function will queue a memory fence to ensure correct
  ordering of memory operations to global memory. This can be useful when work-items,
  for example, write to buffer or image objects and then want to read the updated data.
- dppy.CLK_LOCAL_MEM_FENCE: The barrier function will either flush any variables stored in
  local memory or queue a memory fence to ensure correct ordering of memory operations to
  local memory.

Here's an example of how to use barrier by passing no argument (equivalent to passing global memory fence) in DPPY:

.. literalinclude:: ../../../numba_dppy/examples/barrier.py
   :pyobject: no_arg_barrier_support


Here's an example of how to use barrier by passing a local memory fence in DPPY:

.. literalinclude:: ../../../numba_dppy/examples/barrier.py
   :pyobject: local_memory
