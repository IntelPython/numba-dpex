# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import os

import dpnp
import numpy
import pytest
from numba.tests.support import captured_stdout

from numba_dpex import dpjit, prange
from numba_dpex.core.exceptions import UnsupportedKernelArgumentError


@dpjit
def scale_prange(a, b, scalar):
    dtype = a.dtype

    for i in prange(a.shape[0]):
        c = dtype.type(0.25) * scalar
        b[i] = c * a[i]


def test_unsupported_prange_arg():
    dtype: numpy.dtype = numpy.dtype("float")
    SCALAR = dtype.type(0.5)

    a = dpnp.ones(1024)
    b = dpnp.zeros(1024)

    os.environ["NUMBA_CAPTURED_ERRORS"] = "new_style"
    with pytest.raises(UnsupportedKernelArgumentError):
        scale_prange(a, b, SCALAR)
    os.unsetenv("NUMBA_CAPTURED_ERRORS")
