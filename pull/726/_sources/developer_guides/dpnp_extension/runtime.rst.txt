DPPY runtime
============

Unboxing generates call to ``DPPY_RT_sycl_usm_array_from_python`` which converts
PyObject into native structure for USM array.

This function should be implemented in Python extension.

Extensions are registered in :file:`setup.py`.

Module is written in C or C++.
Module should expose variables with pointers to functions. Cast int values to
pointers to functions.

Functions should be registered in LLVM.

Python module :module:`numba_dppy.runtime._dppy_rt_python`.


Describing extension modules
serup.py
https://docs.python.org/3/distutils/setupscript.html#describing-extension-modules

Extension
https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension

.. code-block:: bash
  python setup.py build_ext


Pyhton/C API
https://docs.python.org/3/c-api/index.html

ctypes for testing
https://docs.python.org/3/library/ctypes.html
