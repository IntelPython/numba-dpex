A *range* kernel represents the simplest form of parallelism that can be
expressed in numba-dpex using kapi. Such a kernel represents a data-parallel
execution over a set of work-items with each work-item representing a logical
thread of execution. :ref:`ex_vecadd_kernel` shows an example of a range kernel
written in numba-dpex.

.. code-block:: python
    :linenos:
    :caption: **Example:** Vector addition using a range kernel
    :name: ex_vecadd_kernel
    :emphasize-lines: 9,17

    import dpnp
    import numba_dpex as dpex
    from numba_dpex import kernel_api as kapi


    # Data parallel kernel implementing vector sum
    @dpex.kernel
    def vecadd(item: kapi.Item, a, b, c):
        i = item.get_id(0)
        c[i] = a[i] + b[i]


    N = 1024
    a = dpnp.ones(N)
    b = dpnp.ones_like(a)
    c = dpnp.zeros_like(a)
    dpex.call_kernel(vecadd, kapi.Range(N), a, b, c)

The highlighted lines in the example demonstrate the definition of the execution
range on **line 17** and extraction of every work-items' *id* or index position
via the ``item.get_id`` call on **line 10**. An execution range comprising of
1024 work-items is defined when calling the kernel and each work-item then
executes a single addition.

There are a few semantic rules that have to be adhered to when writing a range
kernel:

* Analogous to the API of SYCL a range kernel can execute only over a 1-, 2-, or
  a 3-dimensional set of work-items.

* Every range kernel requires its first argument to be an instance of the
  :class:`numba_dpex.kernel_api.Item` class. The ``Item`` object is an
  abstraction encapsulating the index position (id) of a single work-item in the
  global execution range. The id will be a 1-, 2-, or a 3-tuple depending
  the dimensionality of the execution range.

* A range kernel cannot return any value.

  **Note** the rule is enforced only in
  the compiled mode and not in the pure Python execution on a kapi kernel.

* A kernel can accept both array and scalar arguments. Array arguments currently
  can either be a ``dpnp.ndarray`` or a ``dpctl.tensor.usm_ndarray``. Scalar
  values can be of any Python numeric type. Array arguments are passed by
  reference, *i.e.*, changes to an array in a kernel are visible outside the
  kernel. Scalar values are always passed by value.

* At least one argument of a kernel should be an array. The requirement is so
  that the kernel launcher (:func:`numba_dpex.core.kernel_launcher.call_kernel`)
  can determine the execution queue on which to launch the kernel. Refer to the
  :ref:`launching-a-kernel` section for more details.

A range kernel has to be executed via the
:py:func:`numba_dpex.core.kernel_launcher.call_kernel` function by passing in
an instance of the :class:`numba_dpex.kernel_api.Range` class. Refer to the
:ref:`launching-a-kernel` section for more details on how to launch a range
kernel.

A range kernel is meant to express a basic `parallel-for` calculation that is
ideally suited for embarrassingly parallel kernels such as element-wise
computations over n-dimensional arrays (ndarrays). The API for expressing a
range kernel does not allow advanced features such as synchronization of
work-items and fine-grained control over memory allocation on a device. For such
advanced features, an nd-range kernel should be used.

.. seealso::
    Refer API documentation for :class:`numba_dpex.kernel_api.Range`,
    :class:`numba_dpex.kernel_api.Item`,  and
    :func:`numba_dpex.core.kernel_launcher.call_kernel` for more details.
