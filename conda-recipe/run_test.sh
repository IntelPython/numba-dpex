#!/bin/bash

set -e

# For activating OpenCL CPU
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
source ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

set -x

pytest -q -ra --disable-warnings --cov --cov-report term-missing --pyargs numba_dppy -vv

exit 0
