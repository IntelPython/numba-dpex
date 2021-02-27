
Getting Started
===============

Installation
------------

Use setup.py or conda (see conda-recipe).

Testing
-------

See folder numba_dppy/tests.

Run tests
---------

.. code-block:: bash

    python -m pytest --pyargs numba_dppy.tests


Examples
--------

See folder numba_dppy/examples.

To run the examples:

.. code-block:: bash

    python numba_dppy/examples/sum.py

Limitations
-----------

Using ``numba-dppy`` requires requires `Intel Python Numba`_.
Without `Intel Python Numba`_ ``dpctl.device_context`` will have no effect.

.. _`Intel Python Numba`: https://github.com/IntelPython/numba
