cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER "icx")
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_VERBOSE_MAKEFILE ON)

project(onemkl-lapack-iface
    VERSION 1.0
    DESCRIPTION "OneMKL LAPACK Interface"
    LANGUAGES CXX)

if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "$ENV{CONDA_PREFIX} is not set.")
endif(NOT DEFINED ENV{CONDA_PREFIX})

# get_cmake_property(_variableNames VARIABLES)
# list(SORT _variableNames)
# foreach(_variableName ${_variableNames})
# message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
set(OBJ_TARGET_NAME "_dpex_lapack_iface")
set(TARGET_NAME "${OBJ_TARGET_NAME}.cpython-39-x86_64-linux-gnu")

add_dependencies(${TARGET_NAME} ${OBJ_TARGET_NAME})

add_library(${OBJ_TARGET_NAME} OBJECT _dpex_lapack_iface.cpp)
add_library(${TARGET_NAME}
    SHARED ${CMAKE_BINARY_DIR}/CMakeFiles/${OBJ_TARGET_NAME}.dir/${OBJ_TARGET_NAME}.o)

set_target_properties(${OBJ_TARGET_NAME}
    PROPERTIES
    CXX_COMPILER_LAUNCHER ${CMAKE_C_COMPILER}
    COMPILE_FLAGS
    -Wno-unused-result
    -Wsign-compare
    -Wall
    -Wformat
    -Wformat-security
    -fwrapv
    -fstack-protector-all
    -ftree-vectorize
    -fPIC
    -fpic
    -fstack-protector-strong
    -fno-plt
    -ffunction-sections
    -pipe
    -march=nocona
    -mtune=haswell
    -isystem $ENV{CONDA_PREFIX}/include
    -DNDEBUG
    -D_FORTIFY_SOURCE=2
    -O2)

target_include_directories(${OBJ_TARGET_NAME}
    PRIVATE
    $ENV{CONDA_PREFIX}/include
    $ENV{CONDA_PREFIX}/include/python3.9
    $ENV{CONDA_PREFIX}/lib/python3.9/site-packages
    $ENV{CONDA_PREFIX}/lib/python3.9/site-packages/numpy/core/include
    $ENV{CONDA_PREFIX}/lib/python3.9/site-packages/dpctl/include
    $ENV{CMPLR_ROOT}/linux/include
    $ENV{MKLROOT}/include)

set_target_properties(${TARGET_NAME}
    PROPERTIES
    CXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER}
    COMPILE_FLAGS
    -shared
    -pipe
    -Wl,-O2
    -Wl,--sort-common
    -Wl,--as-needed
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,--disable-new-dtags
    -Wl,--gc-sections
    -Wl,-rpath,$ENV{CONDA_PREFIX}/lib
    -Wl,-rpath-link,$ENV{CONDA_PREFIX}/lib
    -Wl,--enable-new-dtags,-R$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/dpctl
    -march=nocona
    -mtune=haswell
    -ftree-vectorize
    -fPIC
    -fpic
    -fstack-protector-strong
    -fno-plt
    -ffunction-sections
    -isystem $ENV{CONDA_PREFIX}/include
    -DNDEBUG
    -D_FORTIFY_SOURCE=2
    -O2)

target_link_libraries(${TARGET_NAME}
    -L$ENV{CONDA_PREFIX}/lib
    -L$ENV{CONDA_PREFIX}/lib/python3.9/site-packages/dpctl
    -L$ENV{MKLROOT}/lib/intel64
    -lmkl_sycl
    -lmkl_intel_ilp64
    -lmkl_tbb_thread
    -lmkl_core
    -lsycl
    -lOpenCL
    -lpthread
    -lm
    -ldl)
