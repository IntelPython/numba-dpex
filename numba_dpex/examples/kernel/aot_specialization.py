# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging

import dpctl.tensor as dpt
import numpy as np

import numba_dpex as dpex
from numba_dpex import float32, int64, usm_ndarray
from numba_dpex.core.exceptions import (
    InvalidKernelSpecializationError,
    MissingSpecializationError,
)

# Similar to Numba, numba-dpex supports ahead-of-time (AOT) compilation of
# functions. The following examples demonstrate the feature for
# numba_dpex.kernel and presents usage scenarios and current limitations.

# ------------                 AOT Example 1.                   ------------ #

# Define type specializations using the numba_dpex usm_ndarray data type.
i64arrty = usm_ndarray(int64, 1, "C", usm_type="device", device="0")
f32arrty = usm_ndarray(float32, 1, "C", usm_type="device", device="0")


# specialize a kernel for the i64arrty
@dpex.kernel((i64arrty, i64arrty, i64arrty))
def data_parallel_sum(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


# run the specialized kernel
a = dpt.ones(1024, dtype=dpt.int64, device="0")
b = dpt.ones(1024, dtype=dpt.int64, device="0")
c = dpt.zeros(1024, dtype=dpt.int64, device="0")

data_parallel_sum[
    1024,
](a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.int64)
assert np.array_equal(npc, npc_expected)


# ------------                 AOT Example 2.                   ------------ #

# Multiple signatures can be specified as a list to AOT compile multiple
# versions of the kernel.

# specialize a kernel for the i64arrty
@dpex.kernel([(i64arrty, i64arrty, i64arrty), (f32arrty, f32arrty, f32arrty)])
def data_parallel_sum2(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


# run the i64 specialized kernel
a = dpt.ones(1024, dtype=dpt.int64, device="0")
b = dpt.ones(1024, dtype=dpt.int64, device="0")
c = dpt.zeros(1024, dtype=dpt.int64, device="0")

data_parallel_sum2[
    1024,
](a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.int64)
assert np.array_equal(npc, npc_expected)

# run the f32 specialized kernel
a = dpt.ones(1024, dtype=dpt.float32, device="0")
b = dpt.ones(1024, dtype=dpt.float32, device="0")
c = dpt.zeros(1024, dtype=dpt.float32, device="0")

data_parallel_sum2[
    1024,
](a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.float32)
assert np.array_equal(npc, npc_expected)


# ------------                 AOT Example 3.                   ------------ #

# AOT specialized kernels cannot be jit compiled. Calling a specialized kernel
# with arguments having type different from the specialization will result in
# an MissingSpecializationError.

a = dpt.ones(1024, dtype=dpt.int32)
b = dpt.ones(1024, dtype=dpt.int32)
c = dpt.zeros(1024, dtype=dpt.int32)

try:
    data_parallel_sum[
        1024,
    ](a, b, c)
except MissingSpecializationError as mse:
    print(mse)


# ------------                 AOT Example 4.                   ------------ #

# Numba_dpex does not support NumPy arrays as kernel arguments and all
# array arguments should be inferable as a numba_dpex.types.usm_ndarray. Trying
# to AOT with a NumPy array-based signature will lead to an
# InvalidKernelSpecializationError

try:
    dpex.kernel((int64[::1], int64[::1], int64[::1]))
except InvalidKernelSpecializationError:
    logging.exception()


# ------------                 Limitations                       ------------ #


# Specifying signatures using strings is not yet supported. The limitation is
# due to numba_dpex relying on Numba's sigutils module to parse signatures.
# Sigutils only recognizes Numba types specified as strings.

try:
    dpex.kernel("(i64arrty, i64arrty, i64arrty)")
except NotImplementedError:
    logging.exception()
