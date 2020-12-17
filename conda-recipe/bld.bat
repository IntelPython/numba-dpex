echo "Activating oneAPI compiler environment..."
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
if errorlevel 1 exit 1
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

%PYTHON% setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
