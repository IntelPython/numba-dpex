Memory management
=================

Data transfer
-------------

At the moment, there is no mechanism for the explicit transfer of arrays to
the device and back. Please use usm arrays.

Device arrays
-------------

At the moment, there is no analogue of
``numba.cuda.cudadrv.devicearray.DeviceNDArray``, please use usm arrays.

Pinned memory
-------------

DPPY does not support pinned memory now.

Queue
-----

The queue is used in DPPY as an analogue of the stream in CUDA.
Below is an example of use for GPU:

.. code-block:: python

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:

And CPU:

.. code-block:: python

    if dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:

In CUDA, an equivalent operation is:

.. code-block:: python

    stream = cuda.stream()
    with stream.auto_synchronize():

When the python with context exits, the stream is automatically synchronized.

Local memory
------------

A limited amount of local memory can be allocated on the device to speed up
access to data, when necessary. That memory will be shared (i.e. both readable
and writable) amongst all threads belonging to a given work group. It also
allows threads to cooperate on a given solution.
You can think of it as a manually-managed data cache.

Local memory in SYCL is an analogue of CUDA shared memory.

To go from CUDA to DPPY, replace ``numba.cuda.shared.array`` with
``dppy.local.static_alloc(shape=blocksize, dtype=float32)``.

``dppy.barrier()`` needs to synchronize all threads in the same thread block.
This function implements the same pattern as barriers in traditional
multi-threaded programming: this function waits until all threads in the block
call it, at which point it returns control to all its callers.

To go from CUDA to DPPY, replace ``numba.cuda.syncthreads`` with
``dppy.local.static_alloc(shape=blocksize, dtype=float32)``.

Here's an example of how to use barrier with global mem fence in DPPY:

.. literalinclude:: ../../numba_dppy/examples/barrier.py
   :pyobject: no_arg_barrier_support

And for local memory:

.. literalinclude:: ../../numba_dppy/examples/barrier.py
   :pyobject: local_memory

Private memory
--------------

DPPY does not support private memory now. CUDA analogue is per-thread local
memory.

Constant memory
---------------

DPPY does not support constant memory now.
