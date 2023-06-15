set "ONEAPI_DEVICE_SELECTOR="

for /F "USEBACKQ tokens=* delims=" %%F in (
`python -c "import dpctl; print(\"\n\".join([dev.backend.name+\":\"+dev.device_type.name for dev in dpctl.get_devices() if dev.device_type.name in [\"cpu\",\"gpu\"]]))"`
) do (
    set "ONEAPI_DEVICE_SELECTOR=%%F"

    pytest -q -ra --disable-warnings --pyargs numba_dpex -vv
    IF %ERRORLEVEL% NEQ 0 exit /B 1
)

exit /B 0
