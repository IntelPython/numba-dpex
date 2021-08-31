#!/bin/bash

set -e
set -x

pytest -q -ra --disable-warnings --cov --cov-report term-missing --pyargs numba_dppy -vv

source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
NUMBA_DPPY_LLVM_SPIRV_ROOT=${ONEAPI_ROOT}/compiler/latest/linux/bin/ pytest -q -ra --disable-warnings --pyargs numba_dppy.tests.kernel_tests.test_atomic_op::test_atomic_fp_native -vv

exit 0
