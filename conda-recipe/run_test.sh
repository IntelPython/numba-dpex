#!/bin/bash

set -e

# For activating OpenCL CPU
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
source ${ONEAPI_ROOT}/tbb/latest/env/vars.sh

set -x

python -v -m pytest

exit 0
