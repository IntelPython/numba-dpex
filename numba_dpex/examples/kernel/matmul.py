#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""The example demonstrates a sliding window matrix-matrix multiplication kernel.
"""

import dpctl.tensor as dpt
import numpy as np

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

square_block_side = 2
work_group_size = (square_block_side, square_block_side)
dtype = np.float32


@dpex.kernel
def matmul(
    nditem: kapi.NdItem,
    X,  # IN READ-ONLY    (X_n_rows, n_cols)
    y,  # IN READ-ONLY    (n_cols, y_n_rows),
    X_slm,  # SLM to store a sliding window over X
    Y_slm,  # SLM to store a sliding window over Y
    result,  # OUT        (X_n_rows, y_n_rows)
):
    X_n_rows = X.shape[0]
    Y_n_cols = y.shape[1]
    n_cols = X.shape[1]

    result_row_idx = nditem.get_global_id(0)
    result_col_idx = nditem.get_global_id(1)

    local_row_idx = nditem.get_local_id(0)
    local_col_idx = nditem.get_local_id(1)

    n_blocks_for_cols = n_cols // square_block_side
    if (n_cols % square_block_side) > 0:
        n_blocks_for_cols += 1

    output = dtype(0)

    gr = nditem.get_group()

    for block_idx in range(n_blocks_for_cols):
        X_slm[local_row_idx, local_col_idx] = dtype(0)
        Y_slm[local_row_idx, local_col_idx] = dtype(0)
        if (result_row_idx < X_n_rows) and (
            (local_col_idx + (square_block_side * block_idx)) < n_cols
        ):
            X_slm[local_row_idx, local_col_idx] = X[
                result_row_idx, local_col_idx + (square_block_side * block_idx)
            ]

        if (result_col_idx < Y_n_cols) and (
            (local_row_idx + (square_block_side * block_idx)) < n_cols
        ):
            Y_slm[local_row_idx, local_col_idx] = y[
                local_row_idx + (square_block_side * block_idx), result_col_idx
            ]

        kapi.group_barrier(gr)

        for idx in range(square_block_side):
            output += X_slm[local_row_idx, idx] * Y_slm[idx, local_col_idx]

        kapi.group_barrier(gr)

    if (result_row_idx < X_n_rows) and (result_col_idx < Y_n_cols):
        result[result_row_idx, result_col_idx] = output


def _arange_reshaped(shape, dtype):
    n_items = shape[0] * shape[1]
    return np.arange(n_items, dtype=dtype).reshape(shape)


X = _arange_reshaped((5, 5), dtype)
Y = _arange_reshaped((5, 5), dtype)
X = dpt.asarray(X)
Y = dpt.asarray(Y)
device = X.device.sycl_device
result = dpt.zeros((5, 5), dtype=dtype, device=device)
X_slm = kapi.LocalAccessor(shape=work_group_size, dtype=dtype)
Y_slm = kapi.LocalAccessor(shape=work_group_size, dtype=dtype)

dpex.call_kernel(
    matmul, kapi.NdRange((6, 6), (2, 2)), X, Y, X_slm, Y_slm, result
)

# Expected:
# [[ 150.  160.  170.  180.  190.]
#  [ 400.  435.  470.  505.  540.]
#  [ 650.  710.  770.  830.  890.]
#  [ 900.  985. 1070. 1155. 1240.]
#  [1150. 1260. 1370. 1480. 1590.]]
print(result)
