#!/bin/bash

set -e

check() {
  echo "Run $1 ..."
  (cd numba_dpex/examples/debug && NUMBA_OPT=0 gdb-oneapi -q -command "$1" python) | grep Done
}

run_checks() {
  check commands/function_breakpoint
  check commands/local_variables_0
  check commands/local_variables_1
  check commands/next
  check commands/sheduler_locking
  check commands/stepi
  check commands/stepping
  check commands/step_dppy_func
  check commands/step_sum
  check commands/simple_sum
  check commands/backtrace
  check commands/backtrace_kernel
  check commands/break_func
  check commands/break_file_func
  check commands/break_line_number
  check commands/break_nested_func
  check commands/info_func
}

run_with_device() {
  echo "Run with SYCL_DEVICE_FILTER=$1 ..."
  SYCL_DEVICE_FILTER=$1 run_checks
}

# run_with_device level_zero:gpu:0
run_with_device opencl:gpu:0
# run_with_device opencl:cpu:0

echo Done
