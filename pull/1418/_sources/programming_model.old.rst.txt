.. _programming_model:
.. include:: ./ext_links.txt

Programming Model
=================

In a heterogeneous system there may be **multiple** devices a Python user may
want to engage. For example, it is common for a consumer-grade laptop to feature
an integrated or a discrete GPU alongside a CPU.

To harness their power one needs to know how to answer the following 3 key
questions:

1. How does a Python program recognize available computational devices?
2. How does a Python workload specify computations to be offloaded to selected
   devices?
3. How does a Python application manage data sharing?

Recognizing available devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python package ``dpctl`` answers these questions. All the computational devices
known to the underlying DPC++ runtime can be accessed using
``dpctl.get_devices()``. A specific device of interest `can be selected
<https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/device_selection.html>`__
either using a helper function, e.g. ``dpctl.select_gpu_device()``, or by
passing a filter selector string to ``dpctl.SyclDevice`` constructor.

.. code:: python

   import dpctl

   # select a GPU device. If multiple devices present,
   # let the underlying runtime select from GPUs
   dev_gpu = dpctl.SyclDevice("gpu")
   # select a CPU device
   dev_cpu = dpctl.SyclDevice("cpu")

   # stand-alone function, equivalent to C++
   #   `auto dev = sycl::gpu_selector().select_device();`
   dev_gpu_alt = dpctl.select_gpu_device()
   # stand-alone function, equivalent to C++
   #   `auto dev = sycl::cpu_selector().select_device();`
   dev_cpu_alt = dpctl.select_cpu_device()

A `device object
<https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/devices.html>`__
can be used to query properies of the device, such as its name, vendor, maximal
number of computational units, memory size, etc.

Specifying offload target
~~~~~~~~~~~~~~~~~~~~~~~~~

To answer the second question on the list we need a digression to explain
offloading in oneAPI DPC++ first.

.. note::
   In DPC++, a computation kernel can be specified using generic C++
   programming and then the kernel can be offloaded to any device that is
   supported by an underlying SYCL runtime. The device to which the kernel
   is offloaded is specified using an **execution queue** when *launching
   the kernel*.

   The oneAPI unified programming model brings portability across heterogeneous
   architectures. Another important aspect of the programming model is its
   inherent flexibility that makes it possible to go beyond portability and even
   strive for performance portability. An oneAPI library may be implemented
   using C++ techniques such as template metaprogramming or dynamic polymorphism
   to implement specializations for a generic kernel. If a kernel is implemented
   polymorphically, the specialized implementation will be dispatched based on
   the execution queue specified during kernel launch. The oneMKL library is an
   example of a performance portable oneAPI library.

A computational task is offloaded for execution on a device by submitting it to
DPC++ runtime which inserts the task in a computational graph. Once the device
becomes available the runtime selects a task whose dependencies are met for
execution. The computational graph as well as the device targeted by its tasks
are stored in a `SYCL queue
<https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/queues.html>`__
object. The task submission is therefore always associated with a queue.

Queues can be constructed directly from a device object, or by using a filter
selector string to indicate the device to construct:

.. code:: python

   # construct queue from device object
   q1 = dpctl.SyclQueue(dev_gpu)
   # construct queue using filter selector
   q2 = dpctl.SyclQueue("gpu")

The computational tasks can be stored in an oneAPI native extension in which
case their submission is orchestrated during Python API calls. Let’s consider a
function that offloads an evaluation of a polynomial for every point of a NumPy
array ``X``. Such a function needs to receive a queue object to indicate which
device the computation must be offloaded to:

.. code:: python

   # allocate space for the result
   Y = np.empty_like(X)
   # evaluate polynomial on the device targeted by the queue, Y[i] = p(X[i])
   onapi_ext.offloaded_poly_evaluate(exec_q, X, Y)

Python call to ``onapi_ext.offloaded_poly_evaluate`` applied to NumPy arrays of
double precision floating pointer numbers gets translated to the following
sample C++ code:

.. code:: cpp

   void
   cpp_offloaded_poly_evaluate(
     sycl::queue q, const double *X, double *Y, size_t n) {
       // create buffers from malloc allocations to make data accessible from device
       sycl::buffer<1, double> buf_X(X, n);
       sycl::buffer<1, double> buf_Y(Y, n);

       q.submit([&](sycl::handler &cgh) {
           // create buffer accessors indicating kernel data-flow pattern
           sycl::accessor acc_X(buf_X, cgh, sycl::read_only);
           sycl::accessor acc_Y(buf_Y, cgh, sycl::write_only, sycl::no_init);

           cgh.parallel_for(n,
              // lambda function that gets executed by different work-items with
              // different arguments in parallel
              [=](sycl::id<1> id) {
                 auto x = accX[id];
                 accY[id] = 3.0 + x * (1.0 + x * (-0.5 + 0.3 * x));
              });
       }).wait();

       return;
   }

We refer an interested reader to an excellent and freely available “`Data
Parallel C++ <https://link.springer.com/book/10.1007%2F978-1-4842-5574-2>`__”
book for details of this data parallel C++.

Our package ``numba_dpex`` allows one to write kernels directly in Python.

.. code:: python

   import numba_dpex


   @numba_dpex.kernel
   def numba_dpex_poly(X, Y):
       i = numba_dpex.get_global_id(0)
       x = X[i]
       Y[i] = 3.0 + x * (1.0 + x * (-0.5 + 0.3 * x))

Specifying the execution queue is done using Python context manager:

.. code:: python

   import numpy as np

   X = np.random.randn(10**6)
   Y = np.empty_like(X)

   with dpctl.device_context(q):
       # apply the kernel to elements of X, writing value into Y,
       # while executing using given queue
       numba_dpex_poly[numba_dpex.Range(X.size)](X, Y)

The argument to ``device_context`` can be a queue object, a device object for
which a temporary queue will be created, or a filter selector string. Thus we
could have equally used ``dpctl.device_context(gpu_dev)`` or
``dpctl.device_context("gpu")``.

Note that in this examples data sharing was implicitly managed for us: in the
case of calling a function from a precompiled oneAPI native extension data
sharing was managed by DPC++ runtime, while in the case of using ``numba_dpex``
kernel it was managed during execution of ``__call__`` method.

Data sharing
~~~~~~~~~~~~

Implicit management of data is surely convenient, but its use in an interpreted
code comes at a performance cost. A runtime must implicitly copy data from host
to the device before the kernel execution commences and then copy some (or all)
of it back after the execution completes for every Python API call.

``dpctl`` provides for allocating memory directly accessible to kernels
executing on a device using SYCL’s Unified Shared Memory (`USM
<https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm>`__)
feature. It also implements USM-based ND-array object
``dpctl.tensor.usm_ndarray`` that conforms `array-API standard
<https://data-apis.org/array-api/latest/>`__.

.. code:: python

   import dpctl.tensor as dpt

   # allocate array of doubles using USM-device allocation on GPU device
   X = dpt.arange(0.0, end=1.0, step=1e-6, device="gpu", usm_type="device")
   # allocate array for the output
   Y = dpt.empty_like(X)

   # execution queue is inferred from allocation queues.
   # Kernel is executed on the same device where arrays were allocated
   numba_dpex_poly[X.size, numba_dpex.DEFAULT_LOCAL_SIZE](X, Y)

The execution queue can be unambiguously determined in this case since both
arguments are USM arrays with the same allocation queues and ``X.sycl_queue ==
Y.sycl_queue`` evaluates to ``True``. Should allocation queues be different,
such an inference becomes ambiguous and ``numba_dpex`` raises
``IndeterminateExecutionQueueError`` advising user to explicitly migrate the
data.

Migration can be accomplished either by using ``dpctl.tensor.asarray(X,
device=target_device)`` to create a copy, or by using
``X.to_device(target_device)`` method.

A USM array can be copied back into a NumPy array using ``dpt.asnumpy(Y)`` if
needed.

Compute follows data
~~~~~~~~~~~~~~~~~~~~

Automatic deduction of the execution queue from allocation queues is consistent
with “`local control for data allocation target
<https://data-apis.org/array-api/latest/design_topics/device_support.html>`__”
in the array API standard. User has full control over memory allocation through
three keyword arguments present in all `array creation functions
<https://data-apis.org/array-api/latest/API_specification/creation_functions.html>`__.
For example, consider

.. code:: python

    # TODO

The keyword ``device`` is `mandated by the array API
<https://data-apis.org/array-api/latest/design_topics/device_support.html#syntax-for-device-assignment>`__.
In ``dpctl.tensor`` the allowed values of the keyword are

-  Filter selector string, e.g. ``device="gpu:0"``
-  Existing ``dpctl.SyclDevice`` object, e.g. ``device=dev_gpu``
-  Existing ``dpctl.SyclQueue`` object
-  ``dpctl.tensor.Device`` object instance obtained from an existing USM array,
   e.g. ``device=X.device``

In all cases, an allocation queue object will be constructed as described
`earlier <#specifying-offload-target>`__ and stored in the array instance,
accessible with ``X.sycl_queue``. Instead of using ``device`` keyword, one can
alternatively use ``sycl_queue`` keyword for readability to directly specify a
``dpctl.SyclQueue`` object to be used as the allocation queue.

The rationale for storing the allocation queue in the array is that kernels
submitted to this queue are guaranteed to be able to correctly dereference (i.e.
access) the USM pointer. Array operations that only involve this single USM
array can thus execute on the allocation queue, and the output array can be
allocated on this same allocation queue with the same usm type as the input
array.

.. note::
   Reusing the allocation queue of the input
   array ensures the computational tasks behind the API call can access the
   array without making implicit copies and the output array is allocated
   on the same device as the input.

Compute follows data is the rule prescribing deduction of the execution and the
allocation queue as well as the USM type for the result when multiple USM arrays
are combined. It stipulates that arrays can be combined if and only if their
allocation *queues are the same* as measured by ``==`` operator (i.e.
``X.sycl_queue == Y.sycl_queue`` must evaluate to ``True``). Same queues refer
to the same underlying task graphs and DPC++ schedulers.

An attempt to combine USM arrays with unsame allocation queues raises an
exception advising the user to migrate the data. Migration can be accomplished
either by using ``dpctl.tensor.asarray(X, device=Y.device)`` to create a copy,
or by using ``X.to_device(Y.device)`` method which can sometime do the migration
more efficiently.

.. warning::
   ``dpctl`` and ``numba_dpex`` are both under heavy development. Feel free to file an
   issue on GitHub or reach out on Gitter should you encounter any issues.
