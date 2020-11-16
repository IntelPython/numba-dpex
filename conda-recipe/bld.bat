%PYTHON% setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1

echo "Activating oneAPI compiler environment..."
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
if errorlevel 1 exit 1
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

echo on

set "CC=clang.exe"

%CC% -flto -target spir64-unknown-unknown -c -x cl -emit-llvm -cl-std=CL2.0 -Xclang -finclude-default-header numba/dppl/ocl/atomics/atomic_ops.cl -o numba/dppl/ocl/atomics/atomic_ops.bc
llvm-spirv -o numba/dppl/ocl/atomics/atomic_ops.spir numba/dppl/ocl/atomics/atomic_ops.bc
xcopy numba\dppl\ocl\atomics\atomic_ops.spir %SP_DIR%\numba\dppl\ocl\atomics /E /Y
