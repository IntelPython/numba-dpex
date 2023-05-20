Getting Started
===============

Installation
------------

Numba-dpex depends on following components:

* numba 0.54.* or 0.55.* (`Numba`_)
* dpctl 0.13.* (`Intel Python dpctl`_)
* dpnp 0.10.1 (`Intel Python DPNP`_)
* `dpcpp-llvm-spirv`_ (SPIRV generation from LLVM IR)
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
    conda create -n numba-dpex-env numba-dpex dpnp -c ${ONEAPI_ROOT}/conda_channel

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

    conda install numba-dpex

Build and Install with setuptools
---------------------------------

``setup.py`` requires environment variable ``ONEAPI_ROOT`` and following packages
installed in conda environment:

.. code-block:: bash

    export ONEAPI_ROOT=/opt/intel/oneapi
    conda create -n numba-dpex-env -c ${ONEAPI_ROOT}/conda_channel python=3.7 dpctl dpnp numba spirv-tools dpcpp-llvm-spirv llvmdev cython pytest
    conda activate numba-dpex-env

Activate DPC++ compiler:

.. code-block:: bash

    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh

For installing:

.. code-block:: bash

    python setup.py install

For development:

.. code-block:: bash

    python setup.py develop


Build and Install with docker
---------------------------------

.. code-block:: bash

    docker run --rm -it \
    -v /path/to/numba-dpex/source:/build \
    -v /path/to/dist:/dist
    ghcr.io/intelpython/numba-dpex/builder:0.20.0-py3.10

    python setup.py develop
    python setup.py bdist_wheel
    cp dist/numba_dpex*.whl /dist/

Now you can install numba-dpex wheel in whatever compatible environment with ``pip``.
You will find ``numba_dpex*.whl`` file in the ``/path/to/dist`` location in
your host system.

You can check what dpctl and dpnp is shipped with builder by running ``pip list``.
In case you need another version, consider building ``builder`` target with necessary
build args. Refer to :ref:`Docker <docker>` section for more details.


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

Docker
------

An easy way you can try `numba_dpex` is by using docker.
To try out numba dpex simply run:

.. code-block:: bash

    docker run --rm -it ghcr.io/intelpython/numba-dpex/runtime:0.20.0-py3.10

.. code-block:: python

    import dpctl

    dpctl.lsplatform()

Refer to :ref:`Docker <docker>` section for more options.

.. _`Numba`: https://github.com/numba/numba
.. _`Intel Python Numba`: https://github.com/IntelPython/numba
.. _`Intel Python dpctl`: https://github.com/IntelPython/dpctl
.. _`Intel Python dpnp`: https://github.com/IntelPython/dpnp
.. _`dpcpp-llvm-spirv`: https://github.com/IntelPython/dpcpp-llvm-spirv
.. _`llvmdev`: https://anaconda.org/intel/llvmdev
.. _`spirv-tools`: https://anaconda.org/intel/spirv-tools
.. _`packaging`: https://packaging.pypa.io/
.. _`scipy`: https://anaconda.org/intel/scipy
.. _`cython`: https://cython.org
.. _`pytest`: https://docs.pytest.org
.. _`Intel Distribution for Python`: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-python.html
.. _`anaconda.org/intel`: https://anaconda.org/intel
.. _`Intel oneAPI`: https://software.intel.com/content/www/us/en/develop/tools/oneapi.html
