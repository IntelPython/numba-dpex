#!/bin/bash

set -euxo pipefail

# Intel LLVM must cooperate with compiler and sysroot from conda
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${BUILD_PREFIX}/lib"

echo "--gcc-toolchain=${BUILD_PREFIX} --sysroot=${BUILD_PREFIX}/${HOST}/sysroot -target ${HOST}" > icpx_for_conda.cfg
ICPXCFG="$(pwd)/icpx_for_conda.cfg"
ICXCFG="$(pwd)/icpx_for_conda.cfg"

export ICXCFG
export ICPXCFG

export CC=icx
export CXX=icpx

# new llvm-spirv location
# starting from dpcpp_impl_linux-64=2022.0.0=intel_3610
PATH=$CONDA_PREFIX/bin-llvm:$PATH
export PATH

SKBUILD_ARGS=(-G Ninja -- -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON)

${PYTHON} setup.py install --single-version-externally-managed --record=record.txt "${SKBUILD_ARGS[@]}"

# Build wheel package
WHEELS_BUILD_ARGS=(-p manylinux2014_x86_64 --build-number "$GIT_DESCRIBE_NUMBER")
if [[ -v WHEELS_OUTPUT_FOLDER ]]; then
    $PYTHON setup.py bdist_wheel "${WHEELS_BUILD_ARGS[@]}" "${SKBUILD_ARGS[@]}"
    cp dist/numba_dpex*.whl "${WHEELS_OUTPUT_FOLDER[@]}"
fi
