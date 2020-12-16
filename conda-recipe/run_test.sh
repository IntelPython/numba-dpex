#!/bin/bash

set -e

# For activating OpenCL CPU
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
source ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

set -x

export NUMBA_DEBUG=1
python -m numba.runtests -b -v -m -- numba_dppy.tests.test_usmarray.TestUsmArray.test_numba_usmarray_as_ndarray

exit 0
