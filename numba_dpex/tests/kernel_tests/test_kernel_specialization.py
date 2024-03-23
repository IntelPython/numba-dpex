# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import dpnp
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
from numba_dpex import DpnpNdArray, float32, int64
from numba_dpex.core.exceptions import InvalidKernelSpecializationError
from numba_dpex.core.types.kernel_api.index_space_ids import ItemType
from numba_dpex.kernel_api import Item, Range

i64arrty = DpnpNdArray(ndim=1, dtype=int64, layout="C")
f32arrty = DpnpNdArray(ndim=1, dtype=float32, layout="C")
item_ty = ItemType(ndim=1)

specialized_kernel1 = dpex.kernel((item_ty, i64arrty, i64arrty, i64arrty))
specialized_kernel2 = dpex.kernel(
    [
        (item_ty, i64arrty, i64arrty, i64arrty),
        (item_ty, f32arrty, f32arrty, f32arrty),
    ]
)


def data_parallel_sum(item: Item, a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = item.get_id(0)
    c[i] = a[i] + b[i]


def test_single_specialization():
    """Test if a kernel can be specialized with a single signature."""
    jitkernel = specialized_kernel1(data_parallel_sum)
    assert len(jitkernel.overloads) == 1


def test_multiple_specialization():
    """Test if a kernel can be specialized with multiple signatures."""
    jitkernel = specialized_kernel2(data_parallel_sum)
    assert len(jitkernel.overloads) == 2


def test_invalid_specialization_error():
    """Test if an InvalidKernelSpecializationError is raised when attempting to
    specialize with NumPy arrays.
    """
    specialized_kernel3 = dpex.kernel(
        (item_ty, int64[::1], int64[::1], int64[::1])
    )
    with pytest.raises(InvalidKernelSpecializationError):
        specialized_kernel3(data_parallel_sum)


def test_missing_specialization_error():
    """Test if a MissingSpecializationError is raised when calling a
    specialized kernel with unsupported arguments.
    """
    SIZE = 16
    a = dpnp.ones(SIZE, dtype=dpt.int32)
    b = dpnp.ones(SIZE, dtype=dpt.int32)
    c = dpnp.zeros(SIZE, dtype=dpt.int32)

    with pytest.raises(TypingError):
        data_parallel_sum_specialized = specialized_kernel1(data_parallel_sum)
        dpex.call_kernel(data_parallel_sum_specialized, Range(SIZE), a, b, c)


def test_execution_of_specialized_kernel():
    """Test if the specialized kernel is correctly executed."""
    SIZE = 16

    a = dpnp.ones(SIZE, dtype=dpt.int64)
    b = dpnp.ones(SIZE, dtype=dpt.int64)
    c = dpnp.zeros(SIZE, dtype=dpt.int64)

    data_parallel_sum_specialized = specialized_kernel1(data_parallel_sum)

    dpex.call_kernel(data_parallel_sum_specialized, Range(SIZE), a, b, c)

    npc = dpnp.asnumpy(c)
    import numpy as np

    npc_expected = np.full(SIZE, 2, dtype=np.int64)
    assert np.array_equal(npc, npc_expected)


def test_string_specialization():
    """Test if NotImplementedError is raised when signature is a string"""

    with pytest.raises(NotImplementedError):
        dpex.kernel("(item_ty, i64arrty, i64arrty, i64arrty)")

    with pytest.raises(NotImplementedError):
        dpex.kernel(
            [
                "(item_ty, i64arrty, i64arrty, i64arrty)",
                "(item_ty, f32arrty, f32arrty, f32arrty)",
            ]
        )

    with pytest.raises(ValueError):
        dpex.kernel((i64arrty))
