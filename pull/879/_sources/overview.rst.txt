.. _overview
.. include:: ./ext_links.txt

Overview
========

Data-Parallel Extensions for Numba* (`numba-dpex`_) is a standalone extension
for the `Numba*`_ Python JIT compiler. Numba-dpex adds two new features to
Numba: an architecture-agnostic kernel programming API, and a new compilation
target that adds typing and compilation support for the `dpnp`_ library. Dpnp is
a Python library for numerical computing that provides a data-parallel
reimplementation of `NumPy*`_'s API. Numba-dpex's support for dpnp compilation
is a new way for Numba users to write code in a NumPy-like API that is
already supported by Numba, while at the same time automatically running such code
parallelly on various types of architecture.

Numba-dpex is being developed as part of `Intel AI Analytics Toolkit`_ and is
distributed with the `Intel Distribution for Python*`_. The extension is also
available on Anaconda cloud and as a Docker image on GitHub. Please refer the
:doc:`getting_started` page to learn more.

Main Features
-------------

Portable kernel programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kernel API has a design and API similar to Numba's ``cuda.jit`` module.
However, the API uses the `SYCL*`_ language runtime and as such is extensible to
various hardware types supported by a SYCL runtime. Presently, numba-dpex uses
the `DPC++`_ SYCL runtime and only supports SPIR-V-based OpenCL and `oneAPI
Level Zero`_ devices CPU and GPU devices.

The following vector addition example illustrates the basic features of the
interface.

.. code-block:: python

    import dpnp
    import numba_dpex as dpex


    @dpex.kernel
    def vecadd_kernel(a, b, c):
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]


    a = dpnp.ones(1024, device="gpu")
    b = dpnp.ones(1024, device="gpu")
    c = dpnp.empty_like(a)

    vecadd_kernel[dpex.Range(1024)](a, b, c)
    print(c)

In the above example, we allocated three arrays on a default ``gpu`` device
using the dpnp library. These arrays are then passed as input arguments to the
kernel function. The compilation target and the subsequent execution of the
kernel is determined completely by the input arguments and follow the
"compute-follows-data" programming model as specified in the `Python* Array API
Standard`_. To change the execution target to a CPU, the device keyword needs to
be changed to ``cpu`` when allocating the dpnp arrays. It is also possible to
leave the ``device`` keyword undefined and let the dpnp library select a default
device based on environment flag settings. Refer the
:doc:`user_manual/kernel_programming/index` for further details.

dpnp compilation and offload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numba-dpex extends Numba's type system and compilation pipeline to compile dpnp
functions and expressions in the same way as NumPy. Unlike Numba's NumPy
compilation that is serial by default, numba-dpex always compiles dpnp
expressions into offloadable kernels and executes them in parallel. The feature
is provided using a decorator ``dpjit`` that behaves identically to
``numba.njit(parallel=True)`` with the addition of dpnp compilation and offload.
Offloading by numba-dpex is not just restricted to CPUs and supports all devices
that are presently supported by the kernel API. ``dpjit`` allows using NumPy and
dpnp expressions in the same function. All NumPy compilation and parallelization
is done via the default Numba code-generation pipeline, whereas dpnp expressions
are compiled using the numba-dpex pipeline.

The vector addition example depicted using the kernel API can be easily
expressed in several different ways using ``dpjit``.

.. code-block:: python

    import dpnp
    import numba_dpex as dpex


    @dpex.dpjit
    def vecadd_v1(a, b):
        return a + b


    @dpex.dpjit
    def vecadd_v2(a, b):
        return dpnp.add(a, b)


    @dpex.dpjit
    def vecadd_v3(a, b):
        c = dpnp.empty_like(a)
        for i in prange(a.shape[0]):
            c[i] = a[i] + b[i]
        return c

As with the kernel API example, a ``dpjit`` function if invoked with dpnp
input arguments follows the compute-follows-data programming model. Refer
:doc:`user_manual/dpnp_offload/index` for further details.

Zero-copy interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~


Contributing
------------

Refer the `contributing guide
<https://github.com/IntelPython/numba-dpex/blob/main/CONTRIBUTING>`_ for
information on coding style and standards used in numba-dpex.

License
-------

Numba-dpex is Licensed under Apache License 2.0 that can be found in `LICENSE
<https://github.com/IntelPython/numba-dpex/blob/main/LICENSE>`_. All usage and
contributions to the project are subject to the terms and conditions of this
license.


Along with the kernel programming API an auto-offload feature is also provided.
The feature enables automatic generation of kernels from data-parallel NumPy
library calls and array expressions, Numba ``prange`` loops, and `other
"data-parallel by construction" expressions
<https://numba.pydata.org/numba-doc/latest/user/parallel.html>`_ that Numba is
able to parallelize. Following two examples demonstrate the two ways in which
kernels may be written using numba-dpex.
