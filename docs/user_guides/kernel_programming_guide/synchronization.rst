``dppy.barrier()`` needs to synchronize all threads in the same thread block.
This function implements the same pattern as barriers in traditional
multi-threaded programming: this function waits until all threads in the block
call it, at which point it returns control to all its callers.

To go from CUDA to DPPY, replace ``numba.cuda.syncthreads`` with
``dppy.local.static_alloc(shape=blocksize, dtype=float32)``.

Here's an example of how to use barrier with global mem fence in DPPY:

.. literalinclude:: ../../numba_dppy/examples/barrier.py
   :pyobject: no_arg_barrier_support
