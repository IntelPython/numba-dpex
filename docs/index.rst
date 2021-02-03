.. numba-dppy documentation master file, created by
   sphinx-quickstart on Tue Jan 19 06:51:06 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to numba-dppy's documentation!
======================================

Numba-dppy is a standalone extension to the `Numba
<https://numba.pydata.org/>`_ JIT compiler that adds `SYCL
<https://www.khronos.org/sycl/>`_ programming capabilities to the compiler.
Numba-dppy supports SYCL programming in two modes:

  - Explicit kernel programming mode.
    
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
        
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            sum[20, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

  - Automatic offloading of NumPy data-parallel expressions and 
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

The two aforementioned examples demonstrate the same operation adding two NumPy 
arrays.

Numba-dppy uses the `dpctl <https://intelpython.github.io/dpctl/>`_ library for
its SYCL interface.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   HowTo
   INDEX
   CONTRIBUTING

.. toctree::
   :caption: Programming SYCL devices
   :maxdepth: 2

   dppy/index.rst
   dppy-reference/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
