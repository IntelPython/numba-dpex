.. _overview:
.. include:: ./ext_links.txt

Overview
========

Data Parallel Extension for Numba* (`numba-dpex`_) is a free and open-source
LLVM-based code generator for portable accelerator programming in Python. The
code generator implements a new pseudo-kernel programming domain-specific
language (DSL) called `KAPI` that is modeled after the C++ DSL `SYCL*`_. The
SYCL language is an open standard developed under the Unified Acceleration
Foundation (`UXL`_) as a vendor-agnostic way of programming different types of
data-parallel hardware such as multi-core CPUs, GPUs, and FPGAs. Numba-dpex and
KAPI aim to bring the same vendor-agnostic and standard-compliant programming
model to Python.

Numba-dpex is built on top of the open-source `Numba*`_ JIT compiler that
implements a CPython bytecode parser and code generator to lower the bytecode to
LLVM IR. The Numba* compiler is able to compile a large sub-set of Python and
most of the NumPy library. Numba-dpex uses Numba*'s tooling to implement the
parsing and typing support for the data types and functions defined in the KAPI
DSL. A custom code generator is then used to lower KAPI to a form of LLVM IR
that includes special LLVM instructions that define a low-level data-parallel
kernel API. Thus, a function defined in KAPI is compiled to a data-parallel
kernel that can run on different types of hardware. Currently, compilation of
KAPI is possible for x86 CPU devices, Intel Gen9 integrated GPUs, Intel UHD
integrated GPUs, and Intel discrete GPUs.


The following example shows a pairwise distance matrix computation in KAPI.

.. code-block:: python

    from numba_dpex import kernel_api as kapi
    import math


    def pairwise_distance_kernel(item: kapi.Item, data, distance):
        i = item.get_id(0)
        j = item.get_id(1)

        data_dims = data.shape[1]

        d = data.dtype.type(0.0)
        for k in range(data_dims):
            tmp = data[i, k] - data[j, k]
            d += tmp * tmp

        distance[j, i] = math.sqrt(d)


Skipping over much of the language details, at a high-level the
``pairwise_distance_kernel`` can be viewed as a data-parallel function that gets
executed individually by a set of "work items". That is, each work item runs the
same function for a subset of the elements of the input ``data`` and
``distance`` arrays. For programmers familiar with the CUDA or OpenCL languages,
it is the same programming model that is referred to as Single Program Multiple
Data (SPMD). As Python has no concept of a work item the KAPI function itself is
sequential and needs to be compiled to convert it into a parallel version. The
next example shows the changes to the original script to compile and run the
``pairwise_distance_kernel`` in parallel.

.. code-block:: python

    from numba_dpex import kernel, call_kernel
    import dpnp

    data = dpnp.random.ranf((10000, 3), device="gpu")
    distance = dpnp.empty(shape=(data.shape[0], data.shape[0]), device="gpu")
    exec_range = kapi.Range(data.shape[0], data.shape[0])
    call_kernel(kernel(pairwise_distance_kernel), exec_range, data, distance)

To compile a KAPI function into a data-parallel kernel and run it on a device,
three things need to be done: allocate the arguments to the function on the
device where the function is to execute, compile the function by applying a
numba-dpex decorator, and `launch` or execute the compiled kernel on the device.

Allocating arrays or scalars to be passed to a compiled KAPI function is not
done directly in numba-dpex. Instead, numba-dpex supports passing in
tensors/ndarrays created using either the `dpnp`_ NumPy drop-in replacement
library or the `dpctl`_ SYCl-based Python Array API library. To trigger
compilation, the ``numba_dpex.kernel`` decorator has to be used, and finally to
launch a compiled kernel the ``numba_dpex.call_kernel`` function should be
invoked.

For a more detailed description about programming with numba-dpex, refer
the :doc:`programming_model`, :doc:`user_guide/index` and the
:doc:`autoapi/index` sections of the documentation. To setup numba-dpex and try
it out refer the :doc:`getting_started` section.
