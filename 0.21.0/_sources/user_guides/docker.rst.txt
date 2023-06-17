.. _docker:

Docker
======

Numba dpex now delivers docker support.
Dockerfile is capable of building numba-dpex as well as direct dependencies for it:
dpctl and dpnp.
There are several prebuilt images available: for trying numba_dpex and
for building numba-dpex.

Building
--------

Numba dpex ships with multistage Dockerfile, which means there are
different `targets <https://docs.docker.com/build/building/multi-stage/#stop-at-a-specific-build-stage>`_ available for build. The most useful ones:

- runtime
- runtime-gpu
- numba-dpex-builder-runtime
- numba-dpex-builder-runtime-gpu

To build docker image

.. code-block:: bash

    docker build --target runtime -t numba-dpex:runtime ./


To run docker image

.. code-block:: bash

    docker run -it --rm numba-dpex:runtime

.. note::

    If you are building docker image with gpu support it will calls github api to get
    latest versions of intel gpu drivers. You may face Github API call limits. To avoid
    this, you can pass your github credentials to increase this limit. You can do it
    by providing
    `build args <https://docs.docker.com/engine/reference/commandline/build/#build-arg>`_
    ``GITHUB_USER`` and ``GITHUB_PASSWORD``. You can use
    `access token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`
    instead of the password:

.. note::

    In case you are building docker image behind firewall and your internet access
    requires proxy, you can provide proxy
    `build args <https://docs.docker.com/engine/reference/commandline/build/#build-arg>`_
    ``http_proxy`` and ``https_proxy``. Please note, that these args must be lowercase.

Dockerfile supports different python versions. To select the one you want, simply set
``PYTHON_VERSION`` build arg. By default docker image is based on official python image
based on slim debian, so the requested python version must be from the available python
docker images. In case you want to build on images on custom image you have to pass
``BASE_IMAGE`` environment variable. Be aware that Dockerfile is based on debian so any
base image should be debian based, like debian or ubuntu.

Build arguments that could be useful:

- PYTHON_VERSION
- CR_TAG
- IGC_TAG
- CM_TAG
- L0_TAG
- ONEAPI_VERSION
- DPCTL_GIT_BRANCH
- DPCTL_GIT_URL
- DPNP_GIT_BRANCH
- DPNP_GIT_URL
- NUMBA_DPEX_GIT_BRANCH
- NUMBA_DPEX_GIT_URL
- CMAKE_VERSION
- CMAKE_VERSION_BUILD
- INTEL_NUMPY_VERSION
- INTEL_NUMBA_VERSION
- CYTHON_VERSION
- SCIKIT_BUILD_VERSION
- http_proxy
- https_proxy
- GITHUB_USER
- GITHUB_PASSWORD
- BASE_IMAGE

Refer to Dockerfile to see all available

Running prebuilt images
-----------------------

An easy way you can try ``numba_dpex`` is by using prebuilt images.
There are several prebuilt images available:

- ``runtime`` package that provides runtime experience
.. code-block:: text

    ghcr.io/intelpython/numba-dpex/runtime:<numba_dpex_version>-py<python_version>[-gpu]

- ``builder`` package that provides building experience
.. code-block:: text

    ghcr.io/intelpython/numba-dpex/builder:<numba_dpex_version>-py<python_version>[-gpu]

- you can also see ``stages`` package, but it is used mostly for building stages.
You can use them to build your own docker that is built on top of one of them.

To try out numba dpex simply run:

.. code-block:: bash

    docker run --rm -it ghcr.io/intelpython/numba-dpex/runtime:0.20.0-py3.10 bash

Within the container you can check available devices by running

.. code-block:: bash

    sycl-ls

or

.. code-block:: bash

    python -c "import dpctl; dpctl.lsplatform();".

.. note::

    If you want to enable GPU you need to pass it within container and use ``*-gpu`` tag.

    For passing GPU into container on linux use arguments ``--device=/dev/dri``.
    However if you are using WSL you need to pass
    ``--device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl`` instead.

So, for example, if you want to run numba dpex container with GPU support on WSL:

.. code-block:: bash

    docker run --rm -it \
    --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl \
    ghcr.io/intelpython/numba-dpex/runtime:0.20.0-py3.10-gpu
