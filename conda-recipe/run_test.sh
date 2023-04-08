#!/bin/bash

set -euxo pipefail

pytest -q -ra --disable-warnings --pyargs numba_dpex -vv

export NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1

pytest -q -ra --disable-warnings -vv \
    --pyargs numba_dpex.tests.kernel_tests.test_atomic_op::test_atomic_fp_native

exit 0
