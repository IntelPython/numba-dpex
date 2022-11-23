Limitations
===========

The following functionality is **limited** or **not supported**.

Altering arguments modified in code
-----------------------------------

Altering arguments has limitation. For it to work correctly
arguments should not be modified in code.
See `Numba issue <https://github.com/numba/numba/pull/7196>`_.

See :ref:`assignment-to-variables`.

Using Numba's direct ``gdb`` bindings in ``nopython`` mode
----------------------------------------------------------

Using Numba's direct ``gdb`` bindings in ``nopython`` mode is not supported in
numba-dpex.

See `Numba documentation <https://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#using-numba-s-direct-gdb-bindings-in-nopython-mode>`_.
