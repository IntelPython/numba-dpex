#!/bin/bash

set -e

# For activating OpenCL CPU
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
source ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

set -x

python -m numba.runtests -b -v -m -- numba.tests
pytest -q -ra --disable-warnings --pyargs numba_dppy -vv

exit 0
