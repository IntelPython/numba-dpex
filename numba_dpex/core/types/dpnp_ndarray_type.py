# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


from .usm_ndarray_type import USMNdArray


class DpnpNdArray(USMNdArray):
    """
    The Numba type to represent an dpnp.ndarray. The type has the same
    structure as USMNdArray used to represnet dpctl.tensor.usm_ndarray.
    """

    pass
