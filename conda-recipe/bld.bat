%PYTHON% setup.py install --single-version-externally-managed --record=record.txt

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% setup.py bdist_wheel
    if errorlevel 1 exit 1
    copy dist\numba_dppy*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)
