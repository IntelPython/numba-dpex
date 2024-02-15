#!/bin/bash

set -e

check() {
  echo "Run $1 ..."
  python "$1" | grep "$SYCL_DEVICE_FILTER"
  # python $1 | grep Done
}

run_checks() {
  check numba_dpex/examples/atomic_op.py
  check numba_dpex/examples/barrier.py
  check numba_dpex/examples/blacksholes_kernel.py
  check numba_dpex/examples/blacksholes_njit.py
  check numba_dpex/examples/dpex_func.py
  check numba_dpex/examples/dpex_with_context.py
  check numba_dpex/examples/matmul.py
  check numba_dpex/examples/pairwise_distance.py
  check numba_dpex/examples/rand.py
  check numba_dpex/examples/sum2D.py
  check numba_dpex/examples/sum_ndarray.py
  check numba_dpex/examples/sum.py
  check numba_dpex/examples/sum_reduction_ocl.py
  check numba_dpex/examples/sum_reduction.py
  check numba_dpex/examples/sum_reduction_recursive_ocl.py
  # check numba_dpex/examples/usm_ndarray.py  # See https://github.com/IntelPython/numba-dpex/issues/436
  check numba_dpex/examples/vectorize.py

  check numba_dpex/examples/auto_offload_examples/sum-1d.py
  check numba_dpex/examples/auto_offload_examples/sum-2d.py
  check numba_dpex/examples/auto_offload_examples/sum-3d.py
  check numba_dpex/examples/auto_offload_examples/sum-4d.py
  check numba_dpex/examples/auto_offload_examples/sum-5d.py

  check numba_dpex/examples/debug/dpex_func.py
  check numba_dpex/examples/debug/sum.py
}

run_with_device() {
  echo "Run with SYCL_DEVICE_FILTER=$1 ..."
  SYCL_DEVICE_FILTER=$1 run_checks
}

run_with_device level_zero:gpu:0
run_with_device opencl:gpu:0
run_with_device opencl:cpu:0

echo Done
