.. _getting_started:
.. include:: ./ext_links.txt

.. |copy| unicode:: U+000A9

.. |trade| unicode:: U+2122

Getting Started
===============


Installation
------------

Numba-dpex depends on following components:

* numba 0.57.*
* dpctl 0.14.*
* dpnp 0.11.*
* dpcpp-cpp-rt
* dpcpp-llvm-spirv
* spirv-tools

It is recommended to use conda packages from the ``anaconda.org/intel`` channel.

Create conda environment:

.. code-block:: bash

    conda create -n numba-dpex-env numba-dpex dpnp -c ${ONEAPI_ROOT}/conda_channel

Build and Install Conda Package
-------------------------------

Create and activate conda build environment:

.. code-block:: bash

    conda create -n build-env conda-build
    conda activate build-env

.. code-block:: bash

    conda build conda-recipe -c intel -c conda-forge

Install conda package:

.. code-block:: bash

    conda install numba-dpex

Build and Install with setuptools
---------------------------------

.. code-block:: bash

    conda create -n numba-dpex-env dpctl dpnp numba spirv-tools dpcpp-llvm-spirv llvmdev cython pytest -c intel -c conda-forge
    conda activate numba-dpex-env


Testing
-------

See folder ``numba_dpex/tests``.

To run the tests:

.. code-block:: bash

    python -m pytest --pyargs numba_dpex.tests

Examples
--------

See folder ``numba_dpex/examples``.

To run the examples:

.. code-block:: bash

    python numba_dpex/examples/sum.py
