echo "Activating oneAPI compiler environment..."
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
if errorlevel 1 exit 1
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

echo on

%PYTHON% setup.py install --single-version-externally-managed --record=record.txt

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% setup.py bdist_wheel
    if errorlevel 1 exit 1
    copy dist\numba_dppy*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)
