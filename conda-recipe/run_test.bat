REM For activating OpenCL CPU
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"

@echo on

export NUMBA_DEBUG=1

python -m numba.runtests -b -v -m -- numba_dppy.tests.test_usmarray.TestUsmArray.test_numba_usmarray_as_ndarray
IF %ERRORLEVEL% NEQ 0 exit /B 1

exit /B 0
