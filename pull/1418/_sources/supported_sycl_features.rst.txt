.. _sycl-vs-dpex:


SYCL* and numba-dpex Feature Comparison
#######################################

The numba-dpex kernel API is developed with the aim of providing a SYCL*-like
kernel programming features directly in Python. The page provides a summary of
the SYCL* kernel programming features that are currently supported in
numba-dpex's kernel API.

Numba-dpex does not implement wrappers or analogues of SYCL's host-callable
runtime API. Such features are provided by the ``dpctl`` package.

.. list-table:: Ranges and index space identifiers
   :widths: 25 25 50
   :header-rows: 1

   * - SYCL* class
     - numba-dpex class
     - Notes
   * - ``range``
     - :class:`numba_dpex.kernel_api.Range`
     -
   * - ``nd_range``
     - :class:`numba_dpex.kernel_api.NdRange`
     -
   * - ``id``
     -
     - Not directly supported. All functions that return an ``id`` object in
       SYCL have versions in numba-dpex that require the dimension to be
       explicitly specified. Equivalent to ``get_id.get(dim)``.
   * - ``item``
     - :class:`numba_dpex.kernel_api.Item`
     -
   * - ``nd_item``
     - :class:`numba_dpex.kernel_api.NdItem`
     -
   * - ``h_item``
     -
     - Not supported. There is no corresponding API in numba-dpex for
       ``group::parallel_for_work_item`` or ``parallel_for_work_group``.
   * - ``group``
     - :class:`numba_dpex.kernel_api.Group`
     -
   * - ``sub_group``
     -
     - Not supported

.. list-table:: Reduction variables
   :widths: 25 25 50
   :header-rows: 1

   * - SYCL* class
     - numba-dpex class
     - Notes
   * - ``reduction``
     -
     - Not supported
   * - ``reducer``
     -
     - Not supported

.. list-table:: Invoking kernels
   :widths: 25 25 50
   :header-rows: 1

   * - SYCL* function for invoking kernels
     - numba-dpex function for invoking kernels
     - Notes
   * - ``single_task``
     -
     - Not supported
   * - ``parallel_for``
     - :func:`numba_dpex.core.kernel_launcher.call_kernel`
     -


.. list-table:: Synchronization and atomics
   :widths: 25 25 50
   :header-rows: 1

   * - SYCL* feature
     - numba-dpex feature
     - Notes
   * - Accessor classes
     -
     - Not supported. Explicit ``sycl::event`` SYCL* objects exposed as
       ``dpctl.SyclEvent`` Python objects can be used for asynchronous kernel
       invocation using the
       :func:`numba_dpex.core.kernel_launcher.call_kernel_async` function.
   * - ``group_barrier``
     -  :func:`numba_dpex.kernel_api.group_barrier`
     - group_barrier does not support synchronization across a sub-group.
   * - ``atomic_fence``
     -  :func:`numba_dpex.kernel_api.atomic_fence`
     -
   * - ``device_event``
     -
     - Not supported
   * - ``atomic_ref``
     - :class:`numba_dpex.kernel_api.AtomicRef`
     - Atomic references are supported for both global and local memory.

.. list-table:: On-device memory allocation
   :widths: 25 25 50
   :header-rows: 1

   * - SYCL* class
     - numba-dpex class
     - Notes
   * - ``local_accessor``
     - :class:`numba_dpex.kernel_api.LocalAccessor`
     -
   * - ``private_memory``
     -
     - Not supported as there is no corresponding API in numba-dpex for
       ``group::parallel_for_work_item`` or ``parallel_for_work_group``.

       Allocating variables on a work-item's private memory can be done using
       :class:`numba_dpex.kernel_api.PrivateMemory`.
   * - Constant memory
     -
     - SYCL 2020 no longer defines a constant memory region in the device memory
       model specification and as such the feature is not implemented by
       numba-dpex.
   * - Global memory
     -
     - Global memory allocation is not handled by numba-dpex and the kernel
       argument is expected to have allocated memory on a device's global
       memory region using a USM allocators. Such allocators are provided by
       the ``dpctl`` package.

.. list-table:: Group functions
   :widths: 25 25 50
   :header-rows: 1

   * - SYCL* group function
     - numba-dpex function
     - Notes
   * - ``group_broadcast``
     -
     - Not supported
   * - ``group_barrier``
     -  :func:`numba_dpex.kernel_api.group_barrier`
     - group_barrier does not support synchronization across a sub-group.

.. list-table:: Group algorithms
   :widths: 25 25 50
   :header-rows: 1

   * - SYCL* group algorithm
     - numba-dpex function
     - Notes
   * - ``joint_any_of``
     -
     - Not supported
   * - ``joint_all_of``
     -
     - Not supported
   * - ``joint_none_of``
     -
     - Not supported
   * - ``any_of_group``
     -
     - Not supported
   * - ``all_of_group``
     -
     - Not supported
   * - ``none_of_group``
     -
     - Not supported
   * - ``shift_group_left``
     -
     - Not supported
   * - ``shift_group_right``
     -
     - Not supported
   * - ``permute_group_by_xor``
     -
     - Not supported
   * - ``select_from_group``
     -
     - Not supported
   * - ``joint_reduce``
     -
     - Not supported
   * - ``reduce_over_group``
     -
     - Not supported
   * - ``joint_exclusive_scan``
     -
     - Not supported
   * - ``joint_inclusive_scan``
     -
     - Not supported
   * - ``exclusive_scan_over_group``
     -
     - Not supported
   * - ``inclusive_scan_over_group``
     -
     - Not supported

.. list-table:: Math functions
   :widths: 25 25 50
   :header-rows: 1

   * - SYCL* math function category
     - numba-dpex
     - Notes
   * - Math functions
     -
     - Refer the kernel programming guide for list of supported functions.
   * - Half and reduced precision math functions
     -
     - Not supported
