.. _overview:
.. include:: ./ext_links.txt

Overview
========

Data Parallel Extension for Numba* (`numba-dpex`_) is a free and open-source
LLVM-based code generator for portable accelerator programming in Python. The
code generator implements a new kernel programming API (kapi) in pure Python
that is modeled after the API of the C++ embedded domain-specific language
(eDSL) `SYCL*`_. The SYCL eDSL is an open standard developed under the Unified
Acceleration Foundation (`UXL`_) as a vendor-agnostic way of programming
different types of data-parallel hardware such as multi-core CPUs, GPUs, and
FPGAs. Numba-dpex and kapi aim to bring the same vendor-agnostic and
standard-compliant programming model to Python.

Numba-dpex is built on top of the open-source `Numba*`_ JIT compiler that
implements a CPython bytecode parser and code generator to lower the bytecode to
LLVM intermediate representation (IR). The Numba* compiler is able to compile a
large sub-set of Python and most of the NumPy library. Numba-dpex uses Numba*'s
tooling to implement the parsing and the typing support for the data types and
functions defined in kapi. A custom code generator is also introduced to lower
kapi functions to a form of LLVM IR that defined a low-level data-parallel
kernel. Thus, a function written kapi although purely sequential when executed
in Python can be compiled to an actual data-parallel kernel that can run on
different types of hardware. Compilation of kapi is possible for x86
CPU devices, Intel Gen9 integrated GPUs, Intel UHD integrated GPUs, and Intel
discrete GPUs.

The following example presents a pairwise distance matrix computation as written
in kapi. A detailed description of the API and all relevant concepts are dealt
with elsewhere in the documentation, for now the example introduces the core
tenet of the programming model.

.. code-block:: python
    :linenos:

    from numba_dpex import kernel_api as kapi
    import math
    import dpnp


    def pairwise_distance_kernel(item: kapi.Item, data, distance):
        i = item.get_id(0)
        j = item.get_id(1)

        data_dims = data.shape[1]

        d = data.dtype.type(0.0)
        for k in range(data_dims):
            tmp = data[i, k] - data[j, k]
            d += tmp * tmp

        distance[j, i] = math.sqrt(d)


    data = dpnp.random.ranf((10000, 3), device="gpu")
    dist = dpnp.empty(shape=(data.shape[0], data.shape[0]), device="gpu")
    exec_range = kapi.Range(data.shape[0], data.shape[0])
    kapi.call_kernel(kernel(pairwise_distance_kernel), exec_range, data, dist)

The ``pairwise_distance_kernel`` function conceptually defines a data-parallel
function to be executed individually by a set of "work items". That is, each
work item runs the function for a subset of the elements of the input ``data``
and ``distance`` arrays. The ``item`` argument passed to the function identifies
the work item that is executing a specific instance of the function. The set of
work items is defined by the ``exec_range`` object and the ``call_kernel`` call
instructs every work item in ``exec_range`` to execute
``pairwise_distance_kernel`` for a specific subset of the data.

The logical abstraction exposed by kapi is referred to as Single Program
Multiple Data (SPMD) programming model. CUDA or OpenCL programmers will
recognize the programming model exposed by kapi as similar to the one in those
languages. However, as Python has no concept of a work item a kapi function
executes sequentially when invoked from Python. To convert it into a true
data-parallel function, the function has to be first compiled using numba-dpex.
The next example shows the changes to the original script to compile and run the
``pairwise_distance_kernel`` in parallel.

.. code-block:: python
    :linenos:
    :emphasize-lines: 7, 25

    import numba_dpex as dpex

    from numba_dpex import kernel_api as kapi
    import math
    import dpnp


    @dpex.kernel
    def pairwise_distance_kernel(item: kapi.Item, data, distance):
        i = item.get_id(0)
        j = item.get_id(1)

        data_dims = data.shape[1]

        d = data.dtype.type(0.0)
        for k in range(data_dims):
            tmp = data[i, k] - data[j, k]
            d += tmp * tmp

        distance[j, i] = math.sqrt(d)


    data = dpnp.random.ranf((10000, 3), device="gpu")
    dist = dpnp.empty(shape=(data.shape[0], data.shape[0]), device="gpu")
    exec_range = kapi.Range(data.shape[0], data.shape[0])

    dpex.call_kernel(pairwise_distance_kernel, exec_range, data, dist)

To compile a kapi function, the ``call_kernel`` function from kapi has to be
substituted by the one provided in ``numba_dpex`` and the ``kernel`` decorator
has to be added to the kapi function. The actual device for which the function
is compiled and on which it executes is controlled by the input arguments to
``call_kernel``. Allocating the input arguments to be passed to a compiled kapi
function is not done by numba-dpex. Instead, numba-dpex supports passing in
tensors/ndarrays created using either the `dpnp`_ NumPy drop-in replacement
library or the `dpctl`_ SYCl-based Python Array API library. The objects
allocated by these libraries encode the device information for that allocation.
Numba-dpex extracts the information and uses it to compile a kernel for that
specific device and then executes the compiled kernel on it.

For a more detailed description about programming with numba-dpex, refer the
:doc:`programming_model`, :doc:`user_guide/index` and the :doc:`autoapi/index`
sections of the documentation. To setup numba-dpex and try it out refer the
:doc:`getting_started` section.
