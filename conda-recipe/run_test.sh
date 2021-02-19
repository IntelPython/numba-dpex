#!/bin/bash

set -e

# For activating OpenCL CPU
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
source ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

set -x

pycc -h
numba -h
numba -s
python -c "from intel_tester import test_routine; test_routine.test_exec()"

pytest -q -ra --disable-warnings --pyargs numba_dppy -vv

exit 0
