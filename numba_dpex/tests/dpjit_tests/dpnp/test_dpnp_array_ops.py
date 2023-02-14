# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp
import numpy as np
import pytest

from numba_dpex import dpjit
from numba_dpex.tests._helper import (
    dpnp_debug,
    filter_strings,
    is_gen12,
    skip_no_dpnp,
    skip_unsupported_dtype,
    skip_windows,
)
from numba_dpex.tests.njit_tests.dpnp._helper import wrapper_function

pytestmark = skip_no_dpnp

list_of_unary_ops = ["cumsum"]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("filter_str", filter_strings)
@pytest.mark.parametrize("unary_op", list_of_unary_ops)
def test_unary_ops(dtype, filter_str, unary_op):
    skip_unsupported_dtype(filter_str, dtype)

    a = dpnp.arange(10, dtype=dtype, device=filter_str)
    op = wrapper_function("x", f"x.{unary_op}()", globals())

    old_fallback = dpnp.config.__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__
    try:
        dpnp.config.__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__ = 0
        f = dpjit(op)
        actual = f(a)
        expected = op(a)
        np.testing.assert_allclose(
            dpnp.asnumpy(actual), dpnp.asnumpy(expected), rtol=1e-3, atol=0
        )
    finally:
        dpnp.config.__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__ = old_fallback
