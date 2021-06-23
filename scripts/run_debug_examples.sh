#!/bin/bash

set -e

check() {
  echo "Run $1 ..."
  (cd numba_dppy/examples/debug && NUMBA_DPPY_DEBUGINFO=1 gdb-oneapi -q -command $1 python) | grep Done
}

run_checks() {
  check commands/local_variables
  check commands/next
  check commands/sheduler_locking
  check commands/stepi
  check commands/stepping
  check commands/backtrace
  check commands/break_func
  check commands/break_file_func
  check commands/break_line_number
  check commands/break_nested_func
}

run_with_device() {
  echo "Run with SYCL_DEVICE_FILTER=$1 ..."
  SYCL_DEVICE_FILTER=$1 run_checks
}

# run_with_device level_zero:gpu:0
run_with_device opencl:gpu:0
# run_with_device opencl:cpu:0

echo Done
