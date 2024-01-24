@REM new llvm-spirv location
@REM starting from dpcpp_impl_win-64=2022.0.0=intel_3638 location is env\Library\bin-llvm
@REM used BUILD_PREFIX as compiler installed in build section of meta.yml
set "PATH=%BUILD_PREFIX%\Library\bin-llvm;%PATH%"

REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
SET "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

REM Since the 60.0.0 release, setuptools includes a local, vendored copy
REM of distutils (from late copies of CPython) that is enabled by default.
REM It breaks build for Windows, so use distutils from "stdlib" as before.
REM @TODO: remove the setting, once transition to build backend on Windows
REM to cmake is complete.
SET "SETUPTOOLS_USE_DISTUTILS=stdlib"

set "CC=icx"
set "CXX=icx"

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
