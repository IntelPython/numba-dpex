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

project(numba-dpex
    VERSION ${NUMBA_DPEX_VERSION}
    DESCRIPTION "An extension for Numba to add data-parallel offload capability"
)

# Help conda build find path from both host and build env.
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

add_subdirectory(numba_dpex/core/runtime)
