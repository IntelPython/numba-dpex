#!/bin/bash

set -e

check() {
  echo "Run $1 ..."
  python "$1" | grep "$SYCL_DEVICE_FILTER"
  # python $1 | grep Done
}

run_checks() {
  check numba_dppy/examples/atomic_op.py
  check numba_dppy/examples/barrier.py
  check numba_dppy/examples/blacksholes_kernel.py
  check numba_dppy/examples/blacksholes_njit.py
  check numba_dppy/examples/dppy_func.py
  check numba_dppy/examples/dppy_with_context.py
  check numba_dppy/examples/matmul.py
  check numba_dppy/examples/pairwise_distance.py
  check numba_dppy/examples/rand.py
  check numba_dppy/examples/sum2D.py
  check numba_dppy/examples/sum_ndarray.py
  check numba_dppy/examples/sum.py
  check numba_dppy/examples/sum_reduction_ocl.py
  check numba_dppy/examples/sum_reduction.py
  check numba_dppy/examples/sum_reduction_recursive_ocl.py
  # check numba_dppy/examples/usm_ndarray.py  # See https://github.com/IntelPython/numba-dppy/issues/436
  check numba_dppy/examples/vectorize.py

  check numba_dppy/examples/auto_offload_examples/sum-1d.py
  check numba_dppy/examples/auto_offload_examples/sum-2d.py
  check numba_dppy/examples/auto_offload_examples/sum-3d.py
  check numba_dppy/examples/auto_offload_examples/sum-4d.py
  check numba_dppy/examples/auto_offload_examples/sum-5d.py

  check numba_dppy/examples/debug/dppy_func.py
  check numba_dppy/examples/debug/sum.py
}

run_with_device() {
  echo "Run with SYCL_DEVICE_FILTER=$1 ..."
  SYCL_DEVICE_FILTER=$1 run_checks
}

run_with_device level_zero:gpu:0
run_with_device opencl:gpu:0
run_with_device opencl:cpu:0

echo Done
