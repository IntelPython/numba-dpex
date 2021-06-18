.. numba-dppy documentation master file, created by
   sphinx-quickstart on Tue Jan 19 06:51:06 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Numba-dppy's documentation!
======================================

`Numba-dppy <https://github.com/IntelPython/numba-dppy>`_ is a standalone extension to the `Numba
<https://numba.pydata.org/>`_ JIT compiler that adds `SYCL
<https://www.khronos.org/sycl/>`_ programming capabilities to Numba.
Numba-dppy uses `dpctl <https://github.com/IntelPython/dpctl>`_ to support
SYCL features and currently Intel's `DPC++ <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md>`_ is the only SYCL runtime supported by
Numba-dppy.

There are two ways to program SYCL devices using Numba-dppy:

  - An explicit kernel programming mode.

    .. code-block:: python

        import numpy as np
        import numba_dppy, numba_dppy as dppy
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

  - An automatic offload mode for NumPy data-parallel expressions and
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
