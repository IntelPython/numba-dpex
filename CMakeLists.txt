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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)

if(IS_INSTALL)
    install(DIRECTORY numba_dpex
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        FILES_MATCHING PATTERN "*.py")
endif()

add_subdirectory(numba_dpex)
