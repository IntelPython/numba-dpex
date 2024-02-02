# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import pytest

import numba_dpex as ndpx
from numba_dpex.kernel_api import NdRange


# Data parallel kernel implementing vector sum
@ndpx.kernel
def kernel_vector_sum(a, b, c):
    i = ndpx.get_global_id(0)
    c[i] = a[i] + b[i]


@pytest.mark.parametrize(
    "error, ranges",
    [
        (TypeError, ((2, 2), ("a", 1, 1))),
        (TypeError, ((3, 3, 3, 3), (2, 2, 2))),
    ],
)
def test_ndrange_config_error(error, ranges):
    """Test if a exception is raised when calling a ndrange kernel with
    unsupported arguments.
    """

    a = dpt.ones(1024, dtype=dpt.int32)
    b = dpt.ones(1024, dtype=dpt.int32)
    c = dpt.zeros(1024, dtype=dpt.int64)

    with pytest.raises(error):
        range = NdRange(ranges[0], ranges[1])
        ndpx.call_kernel(kernel_vector_sum, range, a, b, c)
