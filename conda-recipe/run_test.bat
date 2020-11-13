python -m numba.runtests -b -v -m -- numba_dppy.tests
IF %ERRORLEVEL% NEQ 0 exit /B 1

exit /B 0
