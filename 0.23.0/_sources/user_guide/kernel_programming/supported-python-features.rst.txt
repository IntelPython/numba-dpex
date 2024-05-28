
A kapi function when run in the purely interpreted mode by the CPython
interpreter is a regular Python function, and as such in theory any Python
feature can be used in the body of the function. In practice, to be
JIT compilable and executable on a device only a subset of Python language
features are supported in a kapi function. The restriction stems from both
limitations in the Numba compiler tooling and also from the device-specific
calling convention and other restrictions applied by a device's ABI.

This section provides a partial support matrix for Python features with respect
to their usage in a kapi function.


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

Unsupported Constructs
----------------------

The following Python constructs are **not supported**:

- Exception handling (``try .. except``, ``try .. finally``)
- Context management (the ``with`` statement)
- Comprehensions (either list, dict, set or generator comprehensions)
- Generator (any ``yield`` statements)
- The ``raise`` statement
- The ``assert`` statement
