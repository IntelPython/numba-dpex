#!/bin/bash

set -euxo pipefail
unset ONEAPI_DEVICE_SELECTOR

for selector in $(python -c "import dpctl; print(\" \".join([dev.backend.name+\":\"+dev.device_type.name for dev in dpctl.get_devices() if dev.device_type.name in [\"cpu\",\"gpu\"]]))")
do
    export "ONEAPI_DEVICE_SELECTOR=$selector"
    unset NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1

    pytest -q -ra --disable-warnings --pyargs numba_dpex -vv

    export NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1

    pytest -q -ra --disable-warnings -vv \
        --pyargs numba_dpex.tests.kernel_tests.test_atomic_op::test_atomic_fp_native
done

exit 0
