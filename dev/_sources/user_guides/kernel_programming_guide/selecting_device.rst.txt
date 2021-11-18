How to select device to offload kernels
=======================================

Numba-dppy supports passing two types of arrays alongside scalars to a @numba_dppy.kernel decorated function. Depending on the array argument users will need to use different method to select the device for computation.

The two types are:

1. numpy.ndarray.
2. Any array with __sycl_usm_array_interface__ (SUAI) attribute.

**Users are not allowed to pass mixed type of arrays to a @numba_dppy.kernel.** For example, if the first array argument to a @numba_dppy.kernel is of type :code:`numpy.ndarray`, the rest of the array argument will also have to of type :code:`numpy.ndarray`.

The following are how users can specify in which device they want to offload their computation.

- :code:`numpy.ndarray`
    Using context manager, :code:`with numba_dppy.offload_to_sycl_device(SYCL_device)`. Please look at method :code:`select_device_ndarray()` in the example below.

- Array with __sycl_usm_array_interface__ attribute
     Numba-dppy supports the Compute Follows Data semantics in this case. Compute Follows Data stipulates that computation must be off-loaded to device where data is resident.

     Expected behavior in different cases:
            - Users are allowed to mix arrays created using equivalent SYCL queues. Where equivalent queues are defined as:

                Two SYCL queues are equivalent if they have the same:
                    1. SYCL context
                    2. SYCL device
                    3. Same queue properties


            - All usm-types are accessible from device. Users can mix arrays with different usm-type as long as they were allocated using the equvalent SYCL queue.


Example
~~~~~~
.. literalinclude:: ../../../numba_dppy/examples/select_device_for_kernel.py
