REM For activating OpenCL CPU
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"

@echo on

pycc -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
numba -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
numba -s
IF %ERRORLEVEL% NEQ 0 exit /B 1
python -c "from intel_tester import test_routine; test_routine.test_exec()"
IF %ERRORLEVEL% NEQ 0 exit /B 1

pytest -q -ra --disable-warnings --pyargs numba_dppy -vv
IF %ERRORLEVEL% NEQ 0 exit /B 1

exit /B 0
