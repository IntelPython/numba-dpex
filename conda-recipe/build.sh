#!/bin/bash

set -euxo pipefail

src="${SRC_DIR}"

if [[ "$(uname -s)-suffix" == "Linux-suffix" ]]; then
    echo "starting repack"

    pushd "${src}/llvm_spirv"
    ${PYTHON} setup.py install --single-version-externally-managed --record=llvm_spirv_record.txt
    cat llvm_spirv_record.txt
    popd

    pushd "${src}/compiler"
    cp bin-llvm/llvm-spirv "$(${PYTHON} -c "import llvm_spirv; print(llvm_spirv.llvm_spirv_path())")"
    popd

    echo "done with repack"
fi


pushd ./
${PYTHON} setup.py install --single-version-externally-managed --record=record.txt
popd

# Build wheel package
if [ "$CONDA_PY" == "36" ]; then
    WHEELS_BUILD_ARGS=(-p manylinux1_x86_64)
else
    WHEELS_BUILD_ARGS=(-p manylinux2014_x86_64)
fi
if [[ -v WHEELS_OUTPUT_FOLDER ]]; then
    $PYTHON setup.py bdist_wheel "${WHEELS_BUILD_ARGS[@]}"
    cp dist/numba_dpex*.whl "${WHEELS_OUTPUT_FOLDER[@]}"
fi
