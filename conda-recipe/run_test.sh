#!/bin/bash

set -euxo pipefail

pytest -q -ra --disable-warnings --pyargs numba_dppy -vv

if [[ -v ONEAPI_ROOT ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${ONEAPI_ROOT}/compiler/latest/env/vars.sh"
    set -u

    export NUMBA_DPPY_LLVM_SPIRV_ROOT="${ONEAPI_ROOT}/compiler/latest/linux/bin"
    echo "Using llvm-spirv from oneAPI"
else
    echo "Using llvm-spirv from dpcpp package in conda testing environment"
fi

export NUMBA_DPPY_ACTIVATE_ATOMICS_FP_NATIVE=1

pytest -q -ra --disable-warnings -vv \
    --pyargs numba_dppy.tests.kernel_tests.test_atomic_op::test_atomic_fp_native

exit 0
