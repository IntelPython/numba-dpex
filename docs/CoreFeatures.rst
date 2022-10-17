.. _core_features:

Code-generation based on a device
=================================

In numba-dpex, kernels are written in a device-agnostic fashion making it
easy to write portable code. A kernel is compiled for the device on which the
kernel is enqueued to be executed. The device is specified using a
``dpctl.device_context`` context manager. In the following example, two versions
of the ``sum`` kernel are compiled, one for a GPU and another for a CPU based on
which context the function was invoked. Numba-dpex supports
OpenCL CPU and GPU devices and Level Zero GPU devices. In future, compilation
support may be extended to other type of SYCL devices that are supported by
DPC++'s runtime.

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

        with dpctl.device_context("level_zero:gpu"):
            sum[20, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

        with dpctl.device_context("opencl:cpu"):
            sum[20, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

Automatic offload of NumPy expressions
======================================

A key distinction between numba-dpex and other the GPU backends in Numba is
the ability to automatically offload specific data-parallel sections of a
Numba ``jit`` function.

.. todo::

    Details and examples to be added.

Controllable Fallback
---------------------

By default, if a section of code cannot be offloaded to the GPU, it is
automatically executed on the CPU and warning is printed. This behavior is only
applicable to ``jit`` functions, auto-offloading of NumPy calls, array
expressions and ``prange`` loops. To disable this functionality and force code
running on GPU set the environment variable ``NUMBA_DPEX_FALLBACK_ON_CPU`` to
false (e.g. ``export NUMBA_DPEX_FALLBACK_ON_CPU=0``). In this case the code is
not automatically offloaded to the CPU and errors occur if any.

Offload Diagnostics
-------------------

Setting the debug environment variable ``NUMBA_DPEX_OFFLOAD_DIAGNOSTICS`` (e.g.
``export NUMBA_DPEX_OFFLOAD_DIAGNOSTICS=1``) provides emission of the parallel
and offload diagnostics information based on produced parallel transforms. The
level of detail depends on the integer value between 1 and 4 that is set to the
environment variable (higher is more detailed). In the "Auto-offloading" section
there is the information on which device (device name) this parfor or kernel was
offloaded.
