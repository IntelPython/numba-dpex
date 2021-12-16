set NUMBA_DPPY_TESTING_SKIP_NO_DPNP=1

pytest -q -ra --disable-warnings --pyargs numba_dppy -vv
IF %ERRORLEVEL% NEQ 0 exit /B 1

exit /B 0
