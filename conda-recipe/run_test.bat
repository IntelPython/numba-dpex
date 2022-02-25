pytest -q -ra --disable-warnings --pyargs numba_dppy -vv
IF %ERRORLEVEL% NEQ 0 exit /B 1

@REM Run selected Numba tests

@REM tests sensitive for output
python -m numba.runtests -b -v -m 4 ^
  numba.tests.test_random.TestRandom.test_numpy_gauss_startup ^
  numba.tests.test_random.TestRandom.test_numpy_random_startup ^
  numba.tests.test_random.TestRandom.test_random_gauss_startup ^
  numba.tests.test_random.TestRandom.test_random_random_startup

@REM tests sensitive for warnings
python -m numba.runtests -b -v -m 4 ^
  numba.tests.test_builtins.TestIsinstanceBuiltin.test_experimental_warning ^
  numba.tests.test_dispatcher.TestCache.test_ctypes ^
  numba.tests.test_linalg.TestProduct.test_contiguity_warnings ^
  numba.tests.test_dispatcher.TestCache.test_big_array ^
  numba.tests.test_typedlist.TestTypedList.test_repr_long_list_ipython

python -m numba.runtests -b -v -m 4 ^
  numba.tests.test_dataflow.TestDataFlow.test_assignments ^
  numba.tests.test_dataflow.TestDataFlow.test_assignments2 ^
  numba.tests.test_dataflow.TestDataFlow.test_chained_compare ^
  numba.tests.test_dataflow.TestDataFlow.test_chained_compare_npm

@REM tests sensitive for precision
python -m numba.runtests -b -v -m 4 ^
  numba.tests.test_mathlib.TestMathLib.test_expm1 ^
  numba.tests.test_mathlib.TestMathLib.test_expm1_npm

exit /B 0
