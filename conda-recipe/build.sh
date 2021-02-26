#!/bin/bash


if [ ! -z "${ONEAPI_ROOT}" ]; then
    source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
else
    echo "DPCPP is needed to build OpenCL kernel. Abort!"
fi

${PYTHON} setup.py install --single-version-externally-managed --record=record.txt

# Build wheel package
if [ "$CONDA_PY" == "36" ]; then
    WHEELS_BUILD_ARGS="-p manylinux1_x86_64"
else
    WHEELS_BUILD_ARGS="-p manylinux2014_x86_64"
fi
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
    $PYTHON setup.py bdist_wheel ${WHEELS_BUILD_ARGS}
    cp dist/numba_dppy*.whl ${WHEELS_OUTPUT_FOLDER}
fi
