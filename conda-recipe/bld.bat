@REM new llvm-spirv location
@REM starting from dpcpp_impl_win-64=2022.0.0=intel_3638 location is env\Library\bin-llvm
@REM used BUILD_PREFIX as compiler installed in build section of meta.yml

pushd %SRC_DIR%\llvm_spirv
%PYTHON% setup.py install --single-version-externally-managed --record=llvm_spirv_record.txt
type llvm_spirv_record.txt
popd

pushd %SRC_DIR%\compiler
%PYTHON% -c "import llvm_spirv; print(llvm_spirv.llvm_spirv_path())" > Output
set /p DIRSTR= < Output
copy bin-llvm\llvm-spirv %DIRSTR%
del Output
popd


%PYTHON% setup.py install --single-version-externally-managed --record=record.txt

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% setup.py bdist_wheel
    if errorlevel 1 exit 1
    copy dist\numba_dpex*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)
