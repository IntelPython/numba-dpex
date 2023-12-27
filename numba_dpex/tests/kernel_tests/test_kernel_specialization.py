# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import pytest

import numba_dpex as dpex
from numba_dpex import float32, int64, usm_ndarray
from numba_dpex.core.exceptions import (
    InvalidKernelSpecializationError,
    MissingSpecializationError,
)
from numba_dpex.core.kernel_interface.indexers import Range

i64arrty = usm_ndarray(ndim=1, dtype=int64, layout="C")
f32arrty = usm_ndarray(ndim=1, dtype=float32, layout="C")

specialized_kernel1 = dpex.kernel((i64arrty, i64arrty, i64arrty))
specialized_kernel2 = dpex.kernel(
    [(i64arrty, i64arrty, i64arrty), (f32arrty, f32arrty, f32arrty)]
)


def data_parallel_sum(a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


def test_single_specialization():
    """Test if a kernel can be specialized with a single signature."""
    jitkernel = specialized_kernel1(data_parallel_sum)
    assert jitkernel._specialization_cache.size() == 1


def test_multiple_specialization():
    """Test if a kernel can be specialized with multiple signatures."""
    jitkernel = specialized_kernel2(data_parallel_sum)
    assert jitkernel._specialization_cache.size() == 2


def test_invalid_specialization_error():
    """Test if an InvalidKernelSpecializationError is raised when attempting to
    specialize with NumPy arrays.
    """
    specialized_kernel3 = dpex.kernel((int64[::1], int64[::1], int64[::1]))
    with pytest.raises(InvalidKernelSpecializationError):
        specialized_kernel3(data_parallel_sum)


def test_missing_specialization_error():
    """Test if a MissingSpecializationError is raised when calling a
    specialized kernel with unsupported arguments.
    """
    a = dpt.ones(1024, dtype=dpt.int32)
    b = dpt.ones(1024, dtype=dpt.int32)
    c = dpt.zeros(1024, dtype=dpt.int32)

    with pytest.raises(MissingSpecializationError):
        dpex.call_kernel(
            specialized_kernel1(data_parallel_sum), Range(1024), a, b, c
        )


def test_execution_of_specialized_kernel():
    """Test if the specialized kernel is correctly executed."""
    a = dpt.ones(1024, dtype=dpt.int64)
    b = dpt.ones(1024, dtype=dpt.int64)
    c = dpt.zeros(1024, dtype=dpt.int64)

    dpex.call_kernel(
        specialized_kernel1(data_parallel_sum), Range(1024), a, b, c
    )

    npc = dpt.asnumpy(c)
    import numpy as np

    npc_expected = np.full(1024, 2, dtype=np.int64)
    assert np.array_equal(npc, npc_expected)


def test_string_specialization():
    """Test if NotImplementedError is raised when signature is a string"""

    with pytest.raises(NotImplementedError):
        dpex.kernel("(i64arrty, i64arrty, i64arrty)")

    with pytest.raises(NotImplementedError):
        dpex.kernel(
            ["(i64arrty, i64arrty, i64arrty)", "(f32arrty, f32arrty, f32arrty)"]
        )

    with pytest.raises(ValueError):
        dpex.kernel((i64arrty))
