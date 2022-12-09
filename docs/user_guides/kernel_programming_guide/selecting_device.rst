Defining the execution queue for a kernel function
==================================================
There are two ways to specify the queue where a kernel is executed. The first
way follows the notion of "compute follows data" (CFD). The second way is for a
programmer to specify the execution queue using a dpctl.device_context context
manager.


In the CFD style of programming kernels, the execution queue is determined based
on the input arguments passed to a kernel function. Currently, numba-dpex's
kernel API only supports array arguments that provide the
:code:`__sycl_usm_array_interface__` (SUAI) attribute for CFD style programming.
The SUAI attribute encodes the queue where the array was defined.


We also allow passing arrays and data types that do not provide SUAI. For such
cases, programmers need to specify the queue using the
:code:`dpctl.device_context` context manager. Do note that the use of
:code:`dpctl.device_context` is deprecated and slotted for removal in some
future release.



**Users are not allowed to pass mixed type of arrays to a ``numba_dpex.kernel``
.** For example, if the first array argument to a ``numba_dpex.kernel`` is of
type :code:`numpy.ndarray`, the rest of the array argument will also have to be
of type :code:`numpy.ndarray`.

The following are how users can specify in which device they want to offload their computation.

- :code:`numpy.ndarray`
    Using context manager, :code:`with dpctl.device_context(SYCL_device)`. Please look at method :code:`select_device_ndarray()` in the example below.

- Array with :code:`__sycl_usm_array_interface__` attribute
     Numba-dpex supports the Compute Follows Data semantics in this case.
     Compute Follows Data stipulates that computation must be off-loaded to
     device where data is resident.

     Expected behavior in different cases:
            - Users are allowed to mix arrays created using equivalent SYCL
              queues. Where equivalent queues are defined as:

                Two SYCL queues are equivalent if they have the same:
                    1. SYCL context
                    2. SYCL device
                    3. Same queue properties


            - All usm-types are accessible from device. Users can mix arrays
              with different usm-type as long as they were allocated using the
              equivalent SYCL queue.


            - Using the context_manager to specify a queue when passing SUAI
              args will have no effect on queue selection and Numba-dpex will
              print out a warning to inform the user.


Example
~~~~~~~
.. literalinclude:: ../../../numba_dpex/examples/select_device_for_kernel.py
