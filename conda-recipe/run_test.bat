REM For activating OpenCL CPU
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"

@echo on

python -m pytest --pyargs numba_dppy.tests
IF %ERRORLEVEL% NEQ 0 exit /B 1

exit /B 0
