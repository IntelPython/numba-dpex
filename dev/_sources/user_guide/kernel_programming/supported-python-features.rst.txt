Supported Python Features inside ``numba_dpex.kernel``
======================================================

This page lists the Python features supported inside a ``numba_dpex.kernel``
function.

Built-in types
--------------

**Supported Types**

- ``int``
- ``float``

**Unsupported Types**

- ``complex``
- ``bool``
- ``None``
- ``tuple``

Built-in functions
------------------

The following built-in functions are supported:

- ``abs()``
- ``float``
- ``int``
- ``len()``
- ``range()``
- ``round()``

Standard library modules
------------------------

The following functions from the math module are supported:

-  ``math.acos()``
-  ``math.asin()``
-  ``math.atan()``
-  ``math.acosh()``
-  ``math.asinh()``
-  ``math.atanh()``
-  ``math.cos()``
-  ``math.sin()``
-  ``math.tan()``
-  ``math.cosh()``
-  ``math.sinh()``
-  ``math.tanh()``
-  ``math.erf()``
-  ``math.erfc()``
-  ``math.exp()``
-  ``math.expm1()``
-  ``math.fabs()``
-  ``math.gamma()``
-  ``math.lgamma()``
-  ``math.log()``
-  ``math.log10()``
-  ``math.log1p()``
-  ``math.sqrt()``
-  ``math.ceil()``
-  ``math.floor()``

The following functions from the operator module are supported:

-  ``operator.add()``
-  ``operator.eq()``
-  ``operator.floordiv()``
-  ``operator.ge()``
-  ``operator.gt()``
-  ``operator.iadd()``
-  ``operator.ifloordiv()``
-  ``operator.imod()``
-  ``operator.imul()``
-  ``operator.ipow()``
-  ``operator.isub()``
-  ``operator.itruediv()``
-  ``operator.le()``
-  ``operator.lshift()``
-  ``operator.lt()``
-  ``operator.mod()``
-  ``operator.mul()``
-  ``operator.ne()``
-  ``operator.neg()``
-  ``operator.not_()``
-  ``operator.or_()``
-  ``operator.pos()``
-  ``operator.pow()``
-  ``operator.sub()``
-  ``operator.truediv()``

Unsupported Constructs
----------------------

The following Python constructs are **not supported**:

- Exception handling (``try .. except``, ``try .. finally``)
- Context management (the ``with`` statement)
- Comprehensions (either list, dict, set or generator comprehensions)
- Generator (any ``yield`` statements)
- The ``raise`` statement
- The ``assert`` statement


NumPy support
-------------

NumPy functions are whole array operations and are not supported within a
``numba_dpex.kernel``.
