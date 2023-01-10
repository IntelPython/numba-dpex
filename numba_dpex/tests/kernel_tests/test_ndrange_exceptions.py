# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
import dpnp
import pytest

import numba_dpex as ndpx
from numba_dpex.core.exceptions import (
    UnmatchedNumberOfRangeDimsError,
    UnsupportedGroupWorkItemSizeError,
)


# Data parallel kernel implementing vector sum
@ndpx.kernel
def kernel_vector_sum(a, b, c):
    i = ndpx.get_global_id(0)
    c[i] = a[i] + b[i]


@pytest.mark.parametrize(
    "error, ndrange",
    [
        (UnmatchedNumberOfRangeDimsError, ((2, 2), (1, 1, 1))),
        (UnsupportedGroupWorkItemSizeError, ((3, 3, 3), (2, 2, 2))),
    ],
)
def test_ndrange_config_error(error, ndrange):
    """Test if a exception is raised when calling a
    ndrange kernel with unspported arguments.
    """
    N = 10

    a = dpnp.random.random(N)
    b = dpnp.random.random(N)
    c = dpnp.ones_like(a)

    with pytest.raises(error):
        kernel_vector_sum[ndrange](a, b, c)
