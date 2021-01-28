#!/bin/bash

${PYTHON} setup.py install --single-version-externally-managed --record=record.txt


if [ ! -z "${ONEAPI_ROOT}" ]; then
    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
    export CC=clang
else
    echo "DPCPP is needed to build OpenCL kernel. Abort!"
fi

${CC} -flto -target spir64-unknown-unknown -c -x cl -emit-llvm -cl-std=CL2.0 -Xclang -finclude-default-header numba_dppy/ocl/atomics/atomic_ops.cl -o numba_dppy/ocl/atomics/atomic_ops.bc
llvm-spirv -o numba_dppy/ocl/atomics/atomic_ops.spir numba_dppy/ocl/atomics/atomic_ops.bc
cp numba_dppy/ocl/atomics/atomic_ops.spir ${SP_DIR}/numba_dppy/ocl/atomics/

# Build wheel package
if [ "$CONDA_PY" == "36" ]; then
    WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
else
    WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
fi
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py bdist_wheel ${WHEELS_BUILD_ARGS}
    cp dist/numba_dppy*.whl ${WHEELS_OUTPUT_FOLDER}
fi
