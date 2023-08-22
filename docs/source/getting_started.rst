.. _getting_started:
.. include:: ./ext_links.txt

.. |copy| unicode:: U+000A9

.. |trade| unicode:: U+2122

Getting Started
===============


Installing pre-built conda packages
-----------------------------------

``numba-dpex`` along with its dependencies can be installed using ``conda``.
It is recommended to use conda packages from the ``anaconda.org/intel`` channel
to get the latest production releases.

.. code-block:: bash

    conda create -n numba-dpex-env                                             \
        numba-dpex dpnp dpctl dpcpp-llvm-spirv                     \
        -c intel -c conda-forge

To try out the bleeding edge, the latest packages built from tip of the main
source trunk can be installed from the ``dppy/label/dev`` conda channel.

.. code-block:: bash

    conda create -n numba-dpex-env                                             \
        numba-dpex dpnp dpctl dpcpp-llvm-spirv                     \
        -c dppy/label/dev -c intel -c conda-forge



Building from source
--------------------

``numba-dpex`` can be built from source using either ``conda-build`` or
``setuptools`` (with ``scikit-build`` backend).

Steps to build using ``conda-build``:

1. Ensure ``conda-build`` is installed in the ``base`` conda environment or
   create a new conda environment with ``conda-build`` installed.

.. code-block:: bash

    conda create -n build-env conda-build
    conda activate build-env

2. Build using the vendored conda recipe

.. code-block:: bash

    conda build conda-recipe -c intel -c conda-forge

3. Install the conda package

.. code-block:: bash

    conda install -c local numba-dpex

Steps to build using ``setup.py``:

As before, a conda environment with all necessary dependencies is the suggested
first step.

.. code-block:: bash

    # Create a conda environment that hass needed dependencies installed
    conda create -n numba-dpex-env                                             \
        scikit-build cmake dpctl dpnp numba dpcpp-llvm-spirv llvmdev pytest    \
        -c intel -c conda-forge
    # Activate the environment
    conda activate numba-dpex-env
    # Clone the numba-dpex repository
    git clone https://github.com/IntelPython/numba-dpex.git
    cd numba-dpex
    python setup.py develop

Building inside Docker
----------------------

A Dockerfile is provided on the GitHub repository to build ``numba-dpex``
as well as its direct dependencies: ``dpctl`` and ``dpnp``. Users can either use
one of the pre-built images on the ``numba-dpex`` GitHub page or use the
bundled Dockerfile to build ``numba-dpex`` from source. Using the Dockerfile
also ensures that all device drivers and runtime libraries are pre-installed.

Building
~~~~~~~~

Numba dpex ships with multistage Dockerfile, which means there are
different `targets <https://docs.docker.com/build/building/multi-stage/#stop-at-a-specific-build-stage>`_
available for build. The most useful ones:

- ``runtime``
- ``runtime-gpu``
- ``numba-dpex-builder-runtime``
- ``numba-dpex-builder-runtime-gpu``

To build docker image

.. code-block:: bash

    docker build --target runtime -t numba-dpex:runtime ./


To run docker image

.. code-block:: bash

    docker run -it --rm numba-dpex:runtime

.. note::

    When trying to build a docker image with Intel GPU support, the Dockerfile
    will attempt to use the GitHub API to get the latest Intel GPU drivers.
    Users may run into an issue related to  Github API call limits. The issue
    can be bypassed by providing valid GitHub credentials using the
    ``GITHUB_USER`` and ``GITHUB_PASSWORD``
    `build args <https://docs.docker.com/engine/reference/commandline/build/#build-arg>`_
    to increase the call limit. A GitHub
    `access token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`_
    can also be used instead of the password.

.. note::

    When building the docker image behind a firewall the proxy server settings
    should be provided using the ``http_proxy`` and ``https_proxy`` build args.
    These build args must be specified in lowercase.

The bundled Dockerfile supports different python versions that can be specified
via the ``PYTHON_VERSION`` build arg. By default, the docker image is based on
the official python image based on slim debian. The requested python version
must be from the available python docker images.

The ``BASE_IMAGE`` build arg can be used to build the docker image from a
custom image. Note that as the Dockerfile is based on debian any custom base
image should be debian-based, like debian or ubuntu.

The list of other build args are as follows. Please refer the Dockerfile to
see currently all available build args.

- ``PYTHON_VERSION``
- ``CR_TAG``
- ``IGC_TAG``
- ``CM_TAG``
- ``L0_TAG``
- ``ONEAPI_VERSION``
- ``DPCTL_GIT_BRANCH``
- ``DPCTL_GIT_URL``
- ``DPNP_GIT_BRANCH``
- ``DPNP_GIT_URL``
- ``NUMBA_DPEX_GIT_BRANCH``
- ``NUMBA_DPEX_GIT_URL``
- ``CMAKE_VERSION``
- ``CMAKE_VERSION_BUILD``
- ``INTEL_NUMPY_VERSION``
- ``INTEL_NUMBA_VERSION``
- ``CYTHON_VERSION``
- ``SCIKIT_BUILD_VERSION``
- ``http_proxy``
- ``https_proxy``
- ``GITHUB_USER``
- ``GITHUB_PASSWORD``
- ``BASE_IMAGE``


Using the pre-built images
~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several pre-built docker images available:

- ``runtime`` package that provides a pre-built environment with ``numba-dpex``
              already installed. It is ideal to quickly setup and try
              ``numba-dpex``.

.. code-block:: text

    ghcr.io/intelpython/numba-dpex/runtime:<numba_dpex_version>-py<python_version>[-gpu]

- ``builder`` package that has all required dependencies pre-installed and is
              ideal for building ``numba-dpex`` from source.

.. code-block:: text

    ghcr.io/intelpython/numba-dpex/builder:<numba_dpex_version>-py<python_version>[-gpu]

- ``stages`` package primarily meant for creating a new docker image that is
             built on top of one of the pre-built images.

After setting up the docker image, to run ``numba-dpex`` the following snippet
can be used.

.. code-block:: bash

    docker run --rm -it ghcr.io/intelpython/numba-dpex/runtime:0.20.0-py3.10 bash

It is advisable to verify the SYCL runtime and driver installation within the
container by either running,

.. code-block:: bash

    sycl-ls

or,

.. code-block:: bash

    python -m dpctl -f

.. note::

    To enable GPU device, the ``device`` argument should be used and one of the
    ``*-gpu`` images should be used.

    For passing GPU into container on linux use arguments ``--device=/dev/dri``.
    However if you are using WSL you need to pass
    ``--device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl`` instead.

For example, to run ``numba-dpex`` with GPU support on WSL:

.. code-block:: bash

    docker run --rm -it \
    --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl \
    ghcr.io/intelpython/numba-dpex/runtime:0.20.0-py3.10-gpu



Testing
-------

``numba-dpex`` uses pytest for unit testing and the following example
shows a way to run the unit tests.

.. code-block:: bash

    python -m pytest --pyargs numba_dpex.tests

Examples
--------

A set of examples on how to use ``numba-dpex`` can be found in
``numba_dpex/examples``.
