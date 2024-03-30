Numba-dpex provides a decorator to express auxiliary device-only functions that
can be called from a kernel or another device function, but are not callable
from the host. This decorator :func:`numba_dpex.core.decorators.device_func` has
no direct analogue in SYCL and primarily is provided to help programmers make
their kapi applications modular. :ref:`ex_device_func1` shows a simple usage of
the ``device_func`` decorator.

.. code-block:: python
    :linenos:
    :caption: **Example:** Basic usage of device_func
    :name: ex_device_func1

    import dpnp

    import numba_dpex as dpex
    from numba_dpex import kernel_api as kapi

    # Array size
    N = 10


    @dpex.device_func
    def a_device_function(a):
        """A device callable function that can be invoked from a kernel or
        another device function.
        """
        return a + 1


    @dpex.kernel
    def a_kernel_function(item: kapi.Item, a, b):
        """Demonstrates calling a device function from a kernel."""
        i = item.get_id(0)
        b[i] = a_device_function(a[i])


    N = 16
    a = dpnp.ones(N, dtype=dpnp.int32)
    b = dpnp.zeros(N, dtype=dpnp.int32)

    dpex.call_kernel(a_kernel_function, dpex.Range(N), a, b)


.. code-block:: python
    :linenos:
    :caption: **Example:** Using kapi functionalities in a device_func
    :name: ex_device_func2

    import dpnp

    import numba_dpex as dpex
    from numba_dpex import kernel_api as kapi


    @dpex.device_func
    def increment_value(nd_item: kapi.NdItem, a):
        """Demonstrates the usage of group_barrier and NdItem usage in a
        device_func.
        """
        i = nd_item.get_global_id(0)

        a[i] += 1
        kapi.group_barrier(nd_item.get_group(), kapi.MemoryScope.DEVICE)

        if i == 0:
            for idx in range(1, a.size):
                a[0] += a[idx]


    @dpex.kernel
    def another_kernel(nd_item: kapi.NdItem, a):
        """The kernel does everything by calling a device_func."""
        increment_value(nd_item, a)


    N = 16
    b = dpnp.ones(N, dtype=dpnp.int32)

    dpex.call_kernel(another_kernel, dpex.NdRange((N,), (N,)), b)


A device function does not require the first argument to be an index space id
class, and unlike a kernel function a device function is allowed to return a
value. All kapi functionality can be used in a ``device_func`` decorated
function and at compilation stage numba-dpex will attempt to inline a
``device_func`` into the kernel where it is used.
