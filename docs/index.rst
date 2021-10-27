.. numba-dppy documentation master file, created by
   sphinx-quickstart on Tue Jan 19 06:51:06 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Numba-dppy's documentation!
======================================

`Numba-dppy <https://github.com/IntelPython/numba-dppy>`_ is an Intel
|reg|-developed extension to the `Numba <https://numba.pydata.org/>`_ JIT
compiler that adds "XPU" programming capabilities to it. The
`XPU vision <https://www.intel.com/content/www/us/en/newsroom/news/xpu-vision-oneapi-server-gpu.html#gs.ervs8m>`_
is to make it extremely easy for programmers to write efficient and portable
code for a mix of architectures across CPUs, GPUs, FPGAs and other
accelerators. To provide XPU programming capabilities, Numba-dppy relies on
`SYCL <https://www.khronos.org/sycl/>`_ that is an industry standard for writing
cross-platform code using standard C++. Using a SYCL runtime library Numba-dppy
can launch data-parallel kernels generated directly from Python bytecode on
supported data-parallel architectures. Currently, support for
SYCL is restricted to Intel's
`DPC++ <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md>`_
via the `dpctl <https://github.com/IntelPython/dpctl>`_ package.
Support for other SYCL runtime libraries may be added in the future.

The main feature of Numba-dppy is to let programmers write data-parallel kernels
directly in Python. Such kernels can be written in two different ways: an
explicit API superficially similar to OpenCL, and an implicit API that generates
kernels from NumPy library calls, Numba's ``prange`` statement, and `other
"data-parallel by construction" expressions <https://numba.pydata.org/numba-doc/latest/user/parallel.html>`_
that Numba is able to parallelize. Following are two examples to demonstrate
the two ways in which kernels may be written in a Numba-dppy program.

  - Defining a data-parallel kernel explicitly in Numba-dppy.

    .. code-block:: python

        import numpy as np
        import numba_dppy as dppy
        import dpctl

        @dppy.kernel
        def sum(a, b, c):
            i = dppy.get_global_id(0)
            c[i] = a[i] + b[i]

        a = np.array(np.random.random(20), dtype=np.float32)
        b = np.array(np.random.random(20), dtype=np.float32)
        c = np.ones_like(a)

        with dpctl.device_context("opencl:gpu"):
            sum[20, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

  - Writing implicitly data-parallel expressions in the fashion of
    `Numba parallel loops <https://numba.pydata.org/numba-doc/dev/user/parallel.html#explicit-parallel-loops>`_.

    .. code-block:: python

        from numba import njit
        import numpy as np
        import dpctl

        @njit
        def f1(a, b):
            c = a + b
            return c

        global_size = 64
        local_size = 32
        N = global_size * local_size
        a = np.ones(N, dtype=np.float32)
        b = np.ones(N, dtype=np.float32)
        with dpctl.device_context("opencl:gpu:0"):
            c = f1(a, b)

.. toctree::
   :maxdepth: 1
   :caption: Core Features

   CoreFeatures

.. toctree::
   :maxdepth: 1
   :caption: User Guides

    Getting Started <user_guides/getting_started>
    Programming SYCL Kernels <user_guides/kernel_programming_guide/index>
    Debugging with GDB <user_guides/debugging/index>
    numba-dppy for numba.cuda Programmers <user_guides/migrating_from_numba_cuda>

.. toctree::
    :maxdepth: 1
    :caption: Developer Guides

    developer_guides/dpnp_integration
    developer_guides/tools


About
=====

Numba-dppy is developed by Intel and is part of the `Intel Distribution for
Python <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-python.html>`_.

Contributing
============

Refer the `contributing guide <https://github.com/IntelPython/numba-dppy/blob/main/CONTRIBUTING>`_
for information on coding style and standards used in Numba-dppy.

License
=======

Numba-dppy is Licensed under Apache License 2.0 that can be found in
`LICENSE <https://github.com/IntelPython/numba-dppy/blob/main/LICENSE>`_.
All usage and contributions to the project are subject to the terms and
conditions of this license.


Indices and tables
==================

.. only:: builder_html

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

.. only:: not builder_html

   * :ref:`modindex`

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
