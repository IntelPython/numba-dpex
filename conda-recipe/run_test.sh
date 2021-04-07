#!/bin/bash

set -e

# For activating OpenCL CPU
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
source ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

set -x

coverage run -m --source=numba_dppy --branch --omit=*/numba_dppy/tests/*,*/numba_dppy/_version.py pytest -q -ra --disable-warnings --pyargs numba_dppy -vv && coverage report -m

exit 0
