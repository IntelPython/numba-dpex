Universal Functions
===================

Numba provides a `set of decorators <https://numba.pydata.org/numba-doc/latest/user/vectorize.html>`_ to create `NumPy universal functions <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_-like
routines that are JIT compiled. Although, a close analog to NumPy universal functions Numba's ``@vectorize`` are not fully compatible with a regular NumPy
ufunc.. Refer `Creating NumPy universal functions`_ for details.


``numba-dppy`` only supports ``numba.vectorize`` decorator and not yet the
``numba.guvectorize`` decorator. Another present limitation is that ``numba-dppy`` ufunc kernels cannot invoke ``numba_dppy.kernel`` functions.
Ongoing work is in progress to address these limitations.

Example 1: Basic Example
------------------------

Full example can be found at ``numba_dppy/examples/vectorize.py``.

.. literalinclude:: ../../../numba_dppy/examples/vectorize.py
   :pyobject: ufunc_kernel

.. literalinclude:: ../../../numba_dppy/examples/vectorize.py
   :pyobject: test_ufunc

Example 2: Calling ``numba.vectorize`` inside a ``numba_dppy.kernel``
---------------------------------------------------------------------

Full example can be found at ``numba_dppy/examples/blacksholes_njit.py``.

.. literalinclude:: ../../../numba_dppy/examples/blacksholes_njit.py
   :pyobject: cndf2

.. note::

    ``numba.cuda`` requires ``target='cuda'`` parameter for ``numba.vectorize``
    and ``numba.guvectorize`` functions. ``numba-dppy`` eschews the ``target`` parameter for ``@vectorize`` and infers the target from the
    ``dpctl.device_context`` in which the ``numba.vectorize`` function is
    called.

Full Examples
-------------

- ``numba_dppy/examples/vectorize.py``
- ``numba_dppy/examples/blacksholes_njit.py``

.. _`Universal functions (ufunc)`: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _`Creating NumPy universal functions`: https://numba.pydata.org/numba-doc/latest/user/vectorize.html
