# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp
import numba as nb
import pytest

from numba_dpex import dpjit


def test_pairwise_distance():
    @dpjit
    def pairwise_distance(X1, X2, D):
        """Naïve pairwise distance impl - take an array representing M points in N
        dimensions, and return the M x M matrix of Euclidean distances

        Args:
            X1 : Set of points
            X2 : Set of points
            D  : Outputted distance matrix
        """
        # Size of inputs
        X1_rows = X1.shape[0]
        X2_rows = X2.shape[0]
        X1_cols = X1.shape[1]

        # TODO: get rid of it once prange supports dtype
        # https://github.com/IntelPython/numba-dpex/issues/1063
        float0 = X1.dtype.type(0.0)

        # Outermost parallel loop over the matrix X1
        for i in nb.prange(X1_rows):
            # Loop over the matrix X2
            for j in range(X2_rows):
                d = float0
                # Compute exclidean distance
                for k in range(X1_cols):
                    tmp = X1[i, k] - X2[j, k]
                    d += tmp * tmp
                # Write computed distance to distance matrix
                D[i, j] = dpnp.sqrt(d)

    q = dpctl.SyclQueue()
    X1 = dpnp.ones((100, 2), sycl_queue=q)
    X2 = dpnp.ones((100, 2), sycl_queue=q)
    D = dpnp.empty((100, 100), sycl_queue=q)

    try:
        pairwise_distance(X1, X2, D)
    except:
        pytest.fail("Failed to compile prange loop for pairwise distance calc")
