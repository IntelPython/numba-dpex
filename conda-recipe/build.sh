#!/bin/bash

set -euxo pipefail

echo "starting repack"
src="${SRC_DIR}"

pushd "${src}/llvm_spirv"
${PYTHON} setup.py install --old-and-unmanageable
popd

pushd "${src}/compiler"
cp bin-llvm/llvm-spirv "$(${PYTHON} -c "import llvm_spirv; print(llvm_spirv.llvm_spirv_path())")"
popd
echo "done with repack"


# starting from dpcpp_impl_linux-64=2022.0.0=intel_3610
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
