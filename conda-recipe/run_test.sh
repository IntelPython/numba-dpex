#!/bin/bash

set -euxo pipefail
unset ONEAPI_DEVICE_SELECTOR

for selector in $(python -c "import dpctl; print(\" \".join([dev.backend.name+\":\"+dev.device_type.name for dev in dpctl.get_devices() if dev.device_type.name in [\"cpu\",\"gpu\"]]))")
do
    ONEAPI_DEVICE_SELECTOR=$selector \
    pytest -q -ra --disable-warnings --pyargs numba_dpex -vv
done

exit 0
