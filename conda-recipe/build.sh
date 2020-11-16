#!/bin/bash

${PYTHON} setup.py install --single-version-externally-managed --record=record.txt


if [ ! -z "${ONEAPI_ROOT}" ]; then
    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
    export CC=clang
else
    echo "DPCPP is needed to build OpenCL kernel. Abort!"
fi

$CC -flto -target spir64-unknown-unknown -c -x cl -emit-llvm -cl-std=CL2.0 -Xclang -finclude-default-header numba/dppl/ocl/atomics/atomic_ops.cl -o numba/dppl/ocl/atomics/atomic_ops.bc
llvm-spirv -o numba/dppl/ocl/atomics/atomic_ops.spir numba/dppl/ocl/atomics/atomic_ops.bc
cp numba/dppl/ocl/atomics/atomic_ops.spir ${SP_DIR}/numba/dppl/ocl/atomics/
