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
REM UPD: Seems to work fine with setuptools 69, so we need to set minimal
REM requirements before removing it.
SET "SETUPTOOLS_USE_DISTUTILS=stdlib"

set "CC=icx"
set "CXX=icx"

set "CMAKE_GENERATOR=Ninja"
:: Make CMake verbose
set "VERBOSE=1"

%PYTHON% -m build -w -n -x
if %ERRORLEVEL% neq 0 exit 1

:: `pip install dist\numpy*.whl` does not work on windows,
:: so use a loop; there's only one wheel in dist/ anyway
for /f %%f in ('dir /b /S .\dist') do (
    %PYTHON% -m wheel tags --remove --build %GIT_DESCRIBE_NUMBER% %%f
    if %ERRORLEVEL% neq 0 exit 1
)

:: wheel file was renamed
for /f %%f in ('dir /b /S .\dist') do (
    %PYTHON% -m pip install %%f
    if %ERRORLEVEL% neq 0 exit 1
)

:: Copy wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    copy dist\numba_dpex*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)

REM Delete artifacts from package
rd /s /q "%PREFIX%\__pycache__"
del "%PREFIX%\setup.py"
del "%PREFIX%\LICENSE"
del "%PREFIX%\README.md"
del "%PREFIX%\MANIFEST.in"
