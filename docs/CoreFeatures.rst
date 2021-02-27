.. _core_features:

Code-generation based on a device
=================================

In ``numba-dppy``, kernels are written in a device-agnostic fashion making it
easy to write portable code. A kernel is compiled for the device on which the
kernel is enqueued to be executed. The device is specified using a
``dpctl.device_context`` context manager. In the following example, two versions
of the ``sum`` kernel are compiled, one for a GPU and another for a CPU based on
which context the function was invoked. Currently, ``numba-dppy`` supports
OpenCL CPU and GPU devices and Level Zero GPU devices. In future, compilation
support may be extended to other type of SYCL devices that are supported by
DPC++'s runtime.

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

        with dpctl.device_context("level_zero:gpu"):
            sum[20, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

        with dpctl.device_context("opencl:cpu"):
            sum[20, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

Automatic offload of NumPy expressions
======================================

A key distinction between ``numba-dppy`` and other the GPU backends in Numba is
the ability to automatically offload specific data-parallel sections of a
Numba ``jit`` function.

.. todo::

    Details and examples to be added.

Controllable Fallback
---------------------

With the default behavior of numba-dppy, if a section of code cannot be
offloaded on the GPU, then it is automatically executed on the CPU and printed a
warning. This behavior only applies to ``jit`` functions and auto-offloading of
NumPy functions, array expressions, and ``prange`` loops.

Setting the debug environment variable ``NUMBA_DPPY_FALLBACK_OPTION``
(e.g. ``export NUMBA_DPPY_FALLBACK_OPTION=0``) enables the code is not
automatically offload to the CPU, and an error occurs. This is necessary in
order to understand at an early stage which parts of the code do not work on
the GPU, and not to wait for the program to execute on the CPU if you don't
need it.

Offload Diagnostics
-------------------

Setting the debug environment variable ``NUMBA_DPPY_OFFLOAD_DIAGNOSTICS``
(e.g. ``export NUMBA_DPPY_OFFLOAD_DIAGNOSTICS=1``) enables the parallel and
offload diagnostics information.

If set to an integer value between 1 and 4 (inclusive) diagnostic information
about parallel transforms undertaken by Numba will be written to STDOUT. The
higher the value set the more detailed the information produced.
.. In the "Auto-offloading" section there is the information on which device
.. (device name) this parfor or kernel was offloaded.
