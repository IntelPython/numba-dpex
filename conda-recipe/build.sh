#!/bin/bash


if [ ! -z "${ONEAPI_ROOT}" ]; then
    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
else
    echo "DPCPP is needed to build OpenCL kernel. Abort!"
fi

${PYTHON} setup.py install --single-version-externally-managed --record=record.txt
