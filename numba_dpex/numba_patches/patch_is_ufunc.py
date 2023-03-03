# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from numba.np.ufunc.dufunc import DUFunc


def _dpex_is_ufunc(func):
    return isinstance(func, (np.ufunc, DUFunc)) or hasattr(
        func, "is_dpnp_ufunc"
    )
