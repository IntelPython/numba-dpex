Getting Started
===============

Installation
------------

Numba-dppy depends on following components:

* numba 0.54.* or 0.55.* (`Numba`_)
* dpctl 0.9.* (`Intel Python dpctl`_)
* dpnp >=0.6.* (optional, `Intel Python DPNP`_)
* `llvm-spirv`_ (SPIRV generation from LLVM IR)
* `llvmdev`_ (LLVM IR generation)
* `spirv-tools`_
* `packaging`_
* `cython`_ (for building)
* `scipy`_ (for testing)
* `pytest`_ (for testing)

It is recommended to use conda packages from `Intel Distribution for Python`_
channel or `anaconda.org/intel`_ channel.
`Intel Distribution for Python`_ is available from `Intel oneAPI`_.

Create conda environment:

.. code-block:: bash

    export ONEAPI_ROOT=/opt/intel/oneapi
    conda create -n numba-dppy-env numba-dppy dpnp -c ${ONEAPI_ROOT}/conda_channel

Build and Install Conda Package
-------------------------------

Create and activate conda build environment:

.. code-block:: bash

    conda create -n build-env conda-build
    conda activate build-env

Set environment variable ``ONEAPI_ROOT`` and build conda package:

.. code-block:: bash

    export ONEAPI_ROOT=/opt/intel/oneapi
    conda build conda-recipe -c ${ONEAPI_ROOT}/conda_channel

Install conda package:

.. code-block:: bash

    conda install numba-dppy

Build and Install with setuptools
---------------------------------

``setup.py`` requires environment variable ``ONEAPI_ROOT`` and following packages
installed in conda environment:

.. code-block:: bash

    export ONEAPI_ROOT=/opt/intel/oneapi
    conda create -n numba-dppy-env -c ${ONEAPI_ROOT}/conda_channel python=3.7 dpctl dpnp numba spirv-tools llvm-spirv llvmdev cython pytest
    conda activate numba-dppy-env

Activate DPC++ compiler:

.. code-block:: bash

    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh

For installing:

.. code-block:: bash

    python setup.py install

For development:

.. code-block:: bash

    python setup.py develop

Testing
-------

See folder ``numba_dppy/tests``.

To run the tests:

.. code-block:: bash

    python -m pytest --pyargs numba_dppy.tests

Examples
--------

See folder ``numba_dppy/examples``.

To run the examples:

.. code-block:: bash

    python numba_dppy/examples/sum.py


.. _`Numba`: https://github.com/numba/numba
.. _`Intel Python Numba`: https://github.com/IntelPython/numba
.. _`Intel Python dpctl`: https://github.com/IntelPython/dpctl
.. _`Intel Python dpnp`: https://github.com/IntelPython/dpnp
.. _`llvm-spirv`: https://anaconda.org/intel/llvm-spirv
.. _`llvmdev`: https://anaconda.org/intel/llvmdev
.. _`spirv-tools`: https://anaconda.org/intel/spirv-tools
.. _`packaging`: https://packaging.pypa.io/
.. _`scipy`: https://anaconda.org/intel/scipy
.. _`cython`: https://cython.org
.. _`pytest`: https://docs.pytest.org
.. _`Intel Distribution for Python`: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-python.html
.. _`anaconda.org/intel`: https://anaconda.org/intel
.. _`Intel oneAPI`: https://software.intel.com/content/www/us/en/develop/tools/oneapi.html
