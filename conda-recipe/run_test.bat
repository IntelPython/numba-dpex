REM For activating OpenCL CPU
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"

@echo on

python -m numba.runtests -b -v -m -- numba.tests
IF %ERRORLEVEL% NEQ 0 exit /B 1
pytest -q -ra --disable-warnings --pyargs numba_dppy -vv
IF %ERRORLEVEL% NEQ 0 exit /B 1

exit /B 0
