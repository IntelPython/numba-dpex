Welcome to numba-dpex's documentation!
======================================

Numba data-parallel extension (`numba-dpex
<https://github.com/IntelPython/numba-dpex>`_) is an Intel |reg|-developed
extension to the `Numba <https://numba.pydata.org/>`_ JIT compiler. The
extension adds kernel programming and automatic offload capabilities to the
Numba compiler. Numba-dpex is part of `Intel oneAPI Base Toolkit
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html>`_
and distributed with the `Intel Distribution for Python*
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html>`_.
The goal of the extension is to make it easy for Python programmers to
write efficient and portable code for a mix of architectures across CPUs, GPUs,
FPGAs and other accelerators.

Numba-dpex provides an API to write data-parallel kernels directly in Python and
compiles the kernels to a lower-level kernels that are executed using a `SYCL
<https://www.khronos.org/sycl/>`_ runtime library. Presently, only Intel's
`DPC++ <https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md>`_
SYCL runtime is supported via the `dpctl
<https://github.com/IntelPython/dpctl>`_ package, and only OpenCL and Level Zero
devices are supported. Support for other SYCL runtime libraries and hardwares
may be added in the future.

Along with the kernel programming API an auto-offload feature is also provided.
The feature enables automatic generation of kernels from data-parallel NumPy
library calls and array expressions, Numba ``prange`` loops, and `other
"data-parallel by construction" expressions
<https://numba.pydata.org/numba-doc/latest/user/parallel.html>`_ that Numba is
able to parallelize. Following two examples demonstrate the two ways in
which kernels may be written using numba-dpex.

  - Defining a data-parallel kernel explicitly.

    .. code-block:: python

        import numpy as np
        import numba_dpex as dpex
        import dpctl


        @dpex.kernel
        def sum(a, b, c):
            i = dpex.get_global_id(0)
            c[i] = a[i] + b[i]


        a = np.array(np.random.random(20), dtype=np.float32)
        b = np.array(np.random.random(20), dtype=np.float32)
        c = np.ones_like(a)

        with dpctl.device_context("opencl:gpu"):
            sum[20, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

  - Writing implicitly data-parallel expressions in the fashion of `Numba
    parallel loops
    <https://numba.pydata.org/numba-doc/dev/user/parallel.html#explicit-parallel-loops>`_.

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
    Direct kernel programming <user_guides/kernel_programming_guide/index>
    Debugging with GDB <user_guides/debugging/index>
    Docker <user_guides/docker>
    numba-dpex for numba.cuda Programmers <user_guides/migrating_from_numba_cuda>

.. toctree::
    :maxdepth: 1
    :caption: Developer Guides

    developer_guides/dpnp_integration
    developer_guides/tools


Contributing
============

Refer the `contributing guide
<https://github.com/IntelPython/numba-dpex/blob/main/CONTRIBUTING.md>`_ for
information on coding style and standards used in numba-dpex.

License
=======

Numba-dpex is Licensed under Apache License 2.0 that can be found in `LICENSE
<https://github.com/IntelPython/numba-dpex/blob/main/LICENSE>`_. All usage and
contributions to the project are subject to the terms and conditions of this
license.


Indices and tables
==================

.. only:: builder_html

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

.. only:: not builder_html

   * :ref:`modindex`

.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
