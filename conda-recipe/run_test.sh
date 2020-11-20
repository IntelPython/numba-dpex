#!/bin/bash

set -ex

# For activating OpenCL CPU
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh

python -m numba.runtests -b -v -m -- numba_dppy.tests

exit 0
