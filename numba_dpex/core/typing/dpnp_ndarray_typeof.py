# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.extending import typeof_impl
from numba.np import numpy_support

import numba_dpex.core.typing.typeof as typeof
from numba_dpex.core.types.dpnp_ndarray_types import DpnpNdarray


# This tells Numba how to create a UsmSharedArrayType when a usmarray is passed
# into a njit function.
@typeof_impl.register(DpnpNdarray)
def typeof_dpnp_ndarray(val, c):

    return typeof.typeof_usm_ndarray(val, c)
