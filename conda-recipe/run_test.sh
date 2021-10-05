#!/bin/bash

set -euxo pipefail

echo GOOOOOOOOOOOOOOOOOOOOOOOD

exit 1

if [[ -v ONEAPI_ROOT ]]; then
    DEBUGGER_VERSION=10.1.2
    DEBUGGER_DIR="${ONEAPI_ROOT}/debugger/${DEBUGGER_VERSION}"
    if [[ -d "${DEBUGGER_DIR}" ]]; then
        # shellcheck disable=SC1091
        . "${DEBUGGER_DIR}/env/vars.sh"
    else
        echo "Debugger (${DEBUGGER_DIR}) not installed."
    fi
fi

PYARGS="-k test_debug_dppy_numba -k dummy"
pytest -q -ra --disable-warnings --pyargs numba_dppy -vv "${PYARGS}"

# PYTEST_ARGS="-q -ra --disable-warnings"
# PYARGS="numba_dppy -vv"
# DEBUGGER_VERSION=10.1.2

# if [ -n "$NUMBA_DPPY_TESTING_GDB_ENABLE" ]; then
#     PYARGS="$PYARGS -k test_debug_dppy_numba"

#     # Activate debugger
#     if [[ -v ONEAPI_ROOT ]]; then
#         set +ux
#         # shellcheck disable=SC1090
#         source "${ONEAPI_ROOT}/debugger/${DEBUGGER_VERSION}/env/vars.sh"
#         set -ux
#     fi
# fi

# # shellcheck disable=SC2086
# pytest $PYTEST_ARGS --pyargs $PYARGS

if [[ -v ONEAPI_ROOT ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${ONEAPI_ROOT}/compiler/latest/env/vars.sh"
    set -u

    export NUMBA_DPPY_LLVM_SPIRV_ROOT="${ONEAPI_ROOT}/compiler/latest/linux/bin"
    echo "Using llvm-spirv from oneAPI"
else
    export NUMBA_DPPY_LLVM_SPIRV_ROOT="${CONDA_PREFIX}/bin"
    echo "Using llvm-spirv from conda environment"
fi

pytest -q -ra --disable-warnings -vv \
    --pyargs numba_dppy.tests.kernel_tests.test_atomic_op::test_atomic_fp_native

exit 0
