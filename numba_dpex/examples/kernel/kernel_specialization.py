# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Demonstrates signature specialization feature to pre-compile a kernel.

As opposed to JIT compilation at first call, a ``kernel`` or ``device_func``
decorated function with signature specialization gets compiled on module
load and is cached in memory. The following examples demonstrate the feature for
the numba_dpex.kernel decorator and presents usage scenarios and current
limitations.
"""
import dpctl.tensor as dpt
import numpy as np

import numba_dpex as ndpx
from numba_dpex import float32, int64, usm_ndarray
from numba_dpex.core.exceptions import (
    InvalidKernelSpecializationError,
    MissingSpecializationError,
)
from numba_dpex.core.types.kernel_api.index_space_ids import ItemType

# ------------                 Example 1.                   ------------ #

# Define type specializations using the numba_dpex usm_ndarray data type.
i64arrty = usm_ndarray(1, "C", int64)
f32arrty = usm_ndarray(1, "C", float32)
# Type specialization for the index space id type
itemty = ItemType(ndim=1)


# specialize a kernel for the i64arrty
specialized_kernel = ndpx.kernel((itemty, i64arrty, i64arrty, i64arrty))


def data_parallel_sum(item, a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = item.get_id(0)
    c[i] = a[i] + b[i]


# pre-compiled kernel
pre_compiled_kernel = specialized_kernel(data_parallel_sum)

# run the specialized kernel
a = dpt.ones(1024, dtype=dpt.int64)
b = dpt.ones(1024, dtype=dpt.int64)
c = dpt.zeros(1024, dtype=dpt.int64)

# Call the pre-compiled kernel
ndpx.call_kernel(pre_compiled_kernel, ndpx.Range(1024), a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.int64)
assert np.array_equal(npc, npc_expected)


# ------------                 Example 2.                   ------------ #

# Multiple signatures can be specified as a list to eager compile multiple
# versions of the kernel.


# specialize a kernel for the i64arrty
specialized_kernels_list = ndpx.kernel(
    [
        (itemty, i64arrty, i64arrty, i64arrty),
        (itemty, f32arrty, f32arrty, f32arrty),
    ]
)


def data_parallel_sum2(item, a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = item.get_id(0)
    c[i] = a[i] + b[i]


# Pre-compile both variants of the kernel
pre_compiled_kernels = specialized_kernels_list(data_parallel_sum2)

# run the i64 specialized kernel
a = dpt.ones(1024, dtype=dpt.int64)
b = dpt.ones(1024, dtype=dpt.int64)
c = dpt.zeros(1024, dtype=dpt.int64)

# Compiler will type match the right variant and call it.
ndpx.call_kernel(pre_compiled_kernels, ndpx.Range(1024), a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.int64)
assert np.array_equal(npc, npc_expected)

# run the f32 specialized kernel
a = dpt.ones(1024, dtype=dpt.float32)
b = dpt.ones(1024, dtype=dpt.float32)
c = dpt.zeros(1024, dtype=dpt.float32)

ndpx.call_kernel(pre_compiled_kernels, ndpx.Range(1024), a, b, c)

npc = dpt.asnumpy(c)
npc_expected = np.full(1024, 2, dtype=np.float32)
assert np.array_equal(npc, npc_expected)


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
