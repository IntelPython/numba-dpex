cmake_minimum_required(VERSION 3.21...3.27 FATAL_ERROR)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif()

project(_dpex_onemkl_lapack_iface
    DESCRIPTION "OneMKL LAPACK Python C extension for numba-dpex runtime."
)

# Set compilers, use icx/icpx
set(CMAKE_C_COMPILER "icx")
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_VERBOSE_MAKEFILE ON)

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

# Check CMAKE_MODULE_PATH
message(STATUS "CMAKE_MODULE_PATH=" "${CMAKE_MODULE_PATH}")

# Add packages
find_package(MKL CONFIG REQUIRED)
find_package(IntelSYCL REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(NumPy REQUIRED)
find_package(Dpctl REQUIRED)

# Includes
include(GNUInstallDirs)
include_directories(${Python_INCLUDE_DIRS})
include_directories(${NumPy_INCLUDE_DIRS})
include_directories(${Numba_INCLUDE_DIRS})
include_directories(${Dpctl_INCLUDE_DIRS})

# Source files, *.c
file(GLOB SOURCES "*.c")

# Link dpctl library path with -L
link_directories(${DPCTL_LIBRARY_PATH})

# Output static library, *.so or *.dll
add_library(${PROJECT_NAME} MODULE ${SOURCES})

# Link the static library to python libraries and DPCTLSyclInterface
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${Python_LIBRARIES})
target_link_libraries(${PROJECT_NAME} DPCTLSyclInterface)

# Build python extension module
python_extension_module(${PROJECT_NAME})

# If IS_DEVELOP, copy back the target into numba_dpex/core/runtime
if(IS_DEVELOP)
    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/numba_dpex/onemkl_lapack_iface/*.so
        ${CMAKE_SOURCE_DIR}/numba_dpex/onemkl_lapack_iface/
    )
endif()

# Install
install(
    TARGETS ${PROJECT_NAME} LIBRARY DESTINATION numba_dpex/onemkl_lapack_iface
)
