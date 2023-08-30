@REM new llvm-spirv location
@REM starting from dpcpp_impl_win-64=2022.0.0=intel_3638 location is env\Library\bin-llvm
@REM used BUILD_PREFIX as compiler installed in build section of meta.yml
set "PATH=%BUILD_PREFIX%\Library\bin-llvm;%PATH%"

set "SKBUILD_ARGS=-G Ninja --"
set "SKBUILD_ARGS=%SKBUILD_ARGS% -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"

%PYTHON% setup.py install --single-version-externally-managed --record=record.txt %SKBUILD_ARGS%

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% setup.py bdist_wheel --build-number %GIT_DESCRIBE_NUMBER% %SKBUILD_ARGS%
    if errorlevel 1 exit 1
    copy dist\numba_dpex*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)
