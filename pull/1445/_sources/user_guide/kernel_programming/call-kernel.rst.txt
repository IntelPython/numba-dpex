.. _launching-a-kernel:

Launching a kernel
==================

A ``kernel`` decorated kapi function produces a ``KernelDispatcher`` object that
is a type of a Numba* `Dispatcher`_ object. However, unlike regular Numba*
Dispatcher objects a ``KernelDispatcher`` object cannot be directly invoked from
either CPython or another compiled Numba* ``jit`` function. To invoke a
``kernel`` decorated function, a programmer has to use the
:func:`numba_dpex.core.kernel_launcher.call_kernel` function.

To invoke a ``KernelDispatcher`` the ``call_kernel`` function requires three
things: the ``KernelDispatcher`` object, the ``Range`` or ``NdRange`` object
over which the kernel is to be executed, and the list of arguments to be passed
to the compiled kernel. Once called with the necessary  arguments, the
``call_kernel`` function does the following main things:

- Compiles the ``KernelDispatcher`` object specializing it for the provided
  argument types.

- `Unboxes`_  the kernel arguments by converting CPython objects into Numba* or
   numba-dpex objects.

- Infer the execution queue on which to submit the kernel from the provided
  kernel arguments. (TODO: Refer compute follows data.)

- Submits the kernel to the execution queue.

- Waits for the execution completion, before returning control back to the
  caller.

.. important::
    Programmers should note the following two things when defining the global or
    local range to launch a kernel.

    * Numba-dpex currently limits the maximum allowed global range size to
      ``2^31-1``. It is due to the capabilities of current OpenCL GPU backends
      that generally do not support more than 32-bit global range sizes. A
      kernel requesting a larger global range than that will not execute and a
      ``dpctl._sycl_queue.SyclKernelSubmitError`` will get raised.

      The Intel dpcpp SYCL compiler does handle greater than 32-bit global
      ranges for GPU backends by wrapping the kernel in a new kernel that has
      each work-item perform multiple invocations of the original kernel in a
      32-bit global range. Such a feature is not yet available in numba-dpex.

    * When launching an nd-range kernel, if the number of work-items for a
      particular dimension of a work-group exceeds the maximum device
      capability, it can result in undefined behavior.

    The maximum allowed work-items for a device can be queried programmatically
    as shown in :ref:`ex_max_work_item`.

    .. code-block:: python
        :linenos:
        :caption: **Example:** Query maximum number of work-items for a device
        :name: ex_max_work_item

        import dpctl
        import math

        d = dpctl.SyclDevice("gpu")
        d.print_device_info()

        max_num_work_items = (
            d.max_work_group_size
            * d.max_work_item_sizes1d[0]
            * d.max_work_item_sizes2d[0]
            * d.max_work_item_sizes3d[0]
        )
        print(max_num_work_items, f"(2^{int(math.log(max_num_work_items, 2))})")

        cpud = dpctl.SyclDevice("cpu")
        cpud.print_device_info()

        max_num_work_items_cpu = (
            cpud.max_work_group_size
            * cpud.max_work_item_sizes1d[0]
            * cpud.max_work_item_sizes2d[0]
            * cpud.max_work_item_sizes3d[0]
        )
        print(max_num_work_items_cpu, f"(2^{int(math.log(max_num_work_items_cpu, 2))})")

    The output for :ref:`ex_max_work_item` on a system with an Intel Gen9 integrated
    graphics processor and a 9th Generation Coffee Lake CPU is shown in
    :ref:`ex_max_work_item_output`.

    .. code-block:: bash
        :caption: **OUTPUT:** Query maximum number of work-items for a device
        :name: ex_max_work_item_output

            Name            Intel(R) UHD Graphics 630 [0x3e98]
            Driver version  1.3.24595
            Vendor          Intel(R) Corporation
            Filter string   level_zero:gpu:0

        4294967296 (2^32)
            Name            Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
            Driver version  2023.16.12.0.12_195853.xmain-hotfix
            Vendor          Intel(R) Corporation
            Filter string   opencl:cpu:0

        4503599627370496 (2^52)


The ``call_kernel`` function can be invoked both from CPython and from another
Numba* compiled function. Note that the ``call_kernel`` function supports only
synchronous execution of kernel and the ``call_kernel_async`` function should be
used for asynchronous mode of kernel execution (refer
:ref:`launching-an-async-kernel`).


.. seealso::

    Refer the API documentation for
    :func:`numba_dpex.core.kernel_launcher.call_kernel` for more details.
