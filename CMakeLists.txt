#[=======================================================================[.rst:
numba_dpex
-----------

A cmake file to compile the ``_dpexrt_python`` Python C extension for
``numba_dpex``. You can build this component locally in-place by invoking these
commands:

.. code-block:: cmake
    ~$ cmake .
    ~$ cmake --build . --verbose

Once compiled, the _dpexrt_python library will be in ``numba_dpex/core/runtime``
folder.

This ``CMakeLists.txt`` file will be used by ``setup.py``.
#]=======================================================================]

cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

message(STATUS "NUMBA_DPEX_VERSION=" "${NUMBA_DPEX_VERSION}")

project(numba-dpex
    DESCRIPTION "An extension for Numba to add data-parallel offload capability"
    VERSION ${NUMBA_DPEX_VERSION}
)

add_subdirectory(numba_dpex)
