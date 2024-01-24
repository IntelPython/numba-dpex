# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import numpy as np

import numba_dpex as ndpx
from numba_dpex import float32, int64, usm_ndarray
from numba_dpex.core.exceptions import (
    InvalidKernelSpecializationError,
    MissingSpecializationError,
)

# Similar to Numba, numba-ndpx supports eager compilation of functions. The
# following examples demonstrate the feature for numba_ndpx.kernel and presents
# usage scenarios and current limitations.

# ------------                 Example 1.                   ------------ #

# Define type specializations using the numba_ndpx usm_ndarray data type.
i64arrty = usm_ndarray(1, "C", int64)
f32arrty = usm_ndarray(1, "C", float32)


# specialize a kernel for the i64arrty
@ndpx.kernel((i64arrty, i64arrty, i64arrty))
def data_parallel_sum(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = ndpx.get_global_id(0)
    c[i] = a[i] + b[i]


# run the specialized kernel
a = dpt.ones(1024, dtype=dpt.int64)
b = dpt.ones(1024, dtype=dpt.int64)
c = dpt.zeros(1024, dtype=dpt.int64)

data_parallel_sum[ndpx.Range(1024)](a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.int64)
assert np.array_equal(npc, npc_expected)


# ------------                 Example 2.                   ------------ #

# Multiple signatures can be specified as a list to eager compile multiple
# versions of the kernel.


# specialize a kernel for the i64arrty
@ndpx.kernel([(i64arrty, i64arrty, i64arrty), (f32arrty, f32arrty, f32arrty)])
def data_parallel_sum2(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = ndpx.get_global_id(0)
    c[i] = a[i] + b[i]


# run the i64 specialized kernel
a = dpt.ones(1024, dtype=dpt.int64)
b = dpt.ones(1024, dtype=dpt.int64)
c = dpt.zeros(1024, dtype=dpt.int64)

data_parallel_sum2[ndpx.Range(1024)](a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.int64)
assert np.array_equal(npc, npc_expected)

# run the f32 specialized kernel
a = dpt.ones(1024, dtype=dpt.float32)
b = dpt.ones(1024, dtype=dpt.float32)
c = dpt.zeros(1024, dtype=dpt.float32)

data_parallel_sum2[ndpx.Range(1024)](a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.float32)
assert np.array_equal(npc, npc_expected)


# ------------                 Example 3.                   ------------ #

# A specialized kernel cannot be jit compiled. Calling a specialized kernel
# with arguments having type different from the specialization will result in
# an MissingSpecializationError.

a = dpt.ones(1024, dtype=dpt.int32)
b = dpt.ones(1024, dtype=dpt.int32)
c = dpt.zeros(1024, dtype=dpt.int32)

try:
    data_parallel_sum[ndpx.Range(1024)](a, b, c)
except MissingSpecializationError as mse:
    print(mse)


# ------------                 Example 4.                   ------------ #

# Numba_ndpx does not support NumPy arrays as kernel arguments and all
# array arguments should be inferable as a numba_ndpx.types.usm_ndarray. Trying
# to eager compile with a NumPy array-based signature will lead to an
# InvalidKernelSpecializationError

try:
    ndpx.kernel((int64[::1], int64[::1], int64[::1]))
except InvalidKernelSpecializationError as e:
    print("Dpex kernels cannot be specialized using NumPy arrays.")
    print(e)


# ------------                 Limitations                       ------------ #


# Specifying signatures using strings is not yet supported. The limitation is
# due to numba_ndpx relying on Numba's sigutils module to parse signatures.
# Sigutils only recognizes Numba types specified as strings.

try:
    ndpx.kernel("(i64arrty, i64arrty, i64arrty)")
except NotImplementedError as e:
    print(
        "Dpex kernels cannot be specialized using signatures specified as "
        "strings."
    )
    print(e)

print("Done...")
