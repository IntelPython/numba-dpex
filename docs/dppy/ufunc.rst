DPPY Universal functions
========================

See `Creating NumPy universal functions`_ in Numba for more information about
``@vectorize`` and ``@guvectorize``.

DPPY supports ``@vectorize`` and does not support ``@guvectorize`` yet.

DPPY ``@vectorize`` returns a ufunc-like object which is a close analog
but not fully compatible with a regular NumPy ufunc.

DPPY ufunc kernels don't have the ability to call other DPPY device functions yet.
See :ref:`device-functions`.

Example: Basic Example
----------------------

Full example can be found at ``numba_dppy/examples/vectorize.py``.

.. literalinclude:: ../../numba_dppy/examples/vectorize.py
   :pyobject: ufunc_kernel

.. literalinclude:: ../../numba_dppy/examples/vectorize.py
   :pyobject: test_ufunc

Example: Calling Functions from ``math``
----------------------------------------

Full example can be found at ``numba_dppy/examples/blacksholes_njit.py``.

.. literalinclude:: ../../numba_dppy/examples/blacksholes_njit.py
   :pyobject: cndf2

Transition from Numba CUDA
--------------------------

Numba CUDA requires ``target='cuda'`` parameter for ``@vectorize`` and
``@guvectorize``.
DPPY does not require ``target`` parameter for ``@vectorize``. Just use
``dpctl.device_context`` for running universal function on DPPY devices.

Limitations
-----------

Running universal functions on DPPY devices requires `Intel Python Numba`_.
Without `Intel Python Numba`_ ``dpctl.device_context`` will have no effect.

.. _`Intel Python Numba`: https://github.com/IntelPython/numba


See also:
---------

Examples:

- ``numba_dppy/examples/vectorize.py``
- ``numba_dppy/examples/blacksholes_njit.py``

Theory:

- `Universal functions (ufunc)`_ in NumPy
- `Creating NumPy universal functions`_ in Numba

.. _`Universal functions (ufunc)`: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _`Creating NumPy universal functions`: https://numba.pydata.org/numba-doc/latest/user/vectorize.html
