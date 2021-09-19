#!/bin/bash

set -euxo pipefail

${PYTHON} setup.py install --single-version-externally-managed --record=record.txt

# Build wheel package
if [ "$CONDA_PY" == "36" ]; then
    WHEELS_BUILD_ARGS=(-p manylinux1_x86_64)
else
    WHEELS_BUILD_ARGS=(-p manylinux2014_x86_64)
fi
if [[ -v WHEELS_OUTPUT_FOLDER ]]; then
    $PYTHON setup.py bdist_wheel "${WHEELS_BUILD_ARGS[@]}"
    cp dist/numba_dppy*.whl "${WHEELS_OUTPUT_FOLDER[@]}"
fi
