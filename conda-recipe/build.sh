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

export CMAKE_GENERATOR=Ninja
# Make CMake verbose
export VERBOSE=1

# new llvm-spirv location
# starting from dpcpp_impl_linux-64=2022.0.0=intel_3610
export PATH=$CONDA_PREFIX/bin-llvm:$PATH

# -wnx flags mean: --wheel --no-isolation --skip-dependency-check
${PYTHON} -m build -w -n -x
${PYTHON} -m wheel tags --remove --build "$GIT_DESCRIBE_NUMBER" \
    --platform-tag manylinux2014_x86_64 dist/numba_dpex*.whl
${PYTHON} -m pip install dist/numba_dpex*.whl \
    --no-build-isolation \
    --no-deps \
    --only-binary :all: \
    --no-index \
    --prefix "${PREFIX}" \
    -vv

# Copy wheel package
if [[ -v WHEELS_OUTPUT_FOLDER ]]; then
    cp dist/numba_dpex*.whl "${WHEELS_OUTPUT_FOLDER[@]}"
fi
