#!/bin/bash

THEDIR=$(dirname $(readlink -e ${BASH_SOURCE[0]}))
ROOTDIR=$THEDIR/..

# We can not use common setup script because
# using Intel Python brakes build and run procedure
export ONEAPI_ROOT=/opt/intel/oneapi

. ${ONEAPI_ROOT}/compiler/latest/env/vars.sh

export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest

export PYTHONPATH=$PYTHONPATH:${ROOTDIR}
