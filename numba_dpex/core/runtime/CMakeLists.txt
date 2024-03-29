#[=======================================================================[.rst:
_dpexrt_python
---------------

A cmake file to compile the ``_dpexrt_python`` Python C extension for
``numba_dpex``. You can build this component locally in-place by invoking these
commands:

.. code-block:: cmake
    ~$ cmake .
    ~$ cmake --build . --verbose

Once compiled, the _dpexrt_python library will be in ``numba_dpex/core/runtime``
folder.
#]=======================================================================]

cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif()

project(_dpexrt_python
    DESCRIPTION "A Python C extension for numba-dpex runtime."
)

# Get numba include path
if(NOT DEFINED Numba_INCLUDE_DIRS)
    execute_process(
        COMMAND python -c "import numba; print(numba.extending.include_path());"
        OUTPUT_VARIABLE Numba_INCLUDE_DIRS
        RESULT_VARIABLE RET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(RET EQUAL "1")
        message(FATAL_ERROR "Module \'numba\' not found.")
    endif()
endif()

# Get dpctl library path
if(NOT DEFINED DPCTL_LIBRARY_PATH)
    execute_process(
        COMMAND python -c "import dpctl; import os; print(os.path.dirname(dpctl.__file__));"
        OUTPUT_VARIABLE DPCTL_LIBRARY_PATH
        RESULT_VARIABLE RET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(RET EQUAL "1")
        message(FATAL_ERROR "Module \'dpctl\' not found.")
    endif()
endif()

# Update CMAKE_MODULE_PATH
set(DPCTL_MODULE_PATH ${DPCTL_LIBRARY_PATH}/resources/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_MODULE_PATH})

# Get scikit-build path
if(NOT DEFINED SKBUILD_PATH)
    execute_process(
        COMMAND python -c "import skbuild; print(skbuild.__path__[0]);"
        OUTPUT_VARIABLE SKBUILD_PATH
        RESULT_VARIABLE RET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(RET EQUAL "1")
        message(FATAL_ERROR "Module \'skbuild\' not found.")
    endif()
endif()

# Update CMAKE_MODULE_PATH
set(SKBUILD_MODULE_PATH ${SKBUILD_PATH}/resources/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SKBUILD_MODULE_PATH})

# Add packages
find_package(Python 3.9 REQUIRED
  COMPONENTS Interpreter Development.Module NumPy)
find_package(Dpctl REQUIRED)
find_package(IntelSYCL REQUIRED)

# Source files, *.c
file(GLOB_RECURSE DPEXRT_SOURCES CONFIGURE_DEPENDS "*.c")
file(GLOB_RECURSE KERNEL_SOURCES CONFIGURE_DEPENDS "*.cpp")
set(SOURCES ${DPEXRT_SOURCES} ${KERNEL_SOURCES})

# Link dpctl library path with -L
link_directories(${DPCTL_LIBRARY_PATH})

# Output static library, *.so or *.dll
python_add_library(${PROJECT_NAME} MODULE ${SOURCES})

# Add SYCL to target, this must come after python_add_library()
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${KERNEL_SOURCES})

# Add source headers
target_include_directories(${PROJECT_NAME} PRIVATE .)

# Add Numba Runtime headers
target_include_directories(${PROJECT_NAME} PRIVATE ${Numba_INCLUDE_DIRS})

# Link numpy
target_link_libraries(${PROJECT_NAME} PRIVATE Python::NumPy)

# TODO: switch to CMake, once dpctl supports it.
# Include DPCTL libraray
target_include_directories(${PROJECT_NAME} PRIVATE ${Dpctl_INCLUDE_DIRS})
# Linking target binary with DPCTLSyclInterface.so/.dll so the symbols could
# be found during runtime.
target_link_libraries(${PROJECT_NAME} PRIVATE DPCTLSyclInterface)

# Install
install(
    TARGETS ${PROJECT_NAME} LIBRARY DESTINATION numba_dpex/core/runtime
)
