# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dpnp import ndarray
from numba.core import types

from numba_dpex.core.types import Array


class dpnp_ndarray_Type(Array):
    """Numba type for dpnp.ndarray."""

    def __init__(
        self,
        dtype,
        ndim,
        layout,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=None,
        usm_type=None,
    ):
        if name is None:
            type_name = "dpnp.ndarray"
            if readonly:
                type_name = "readonly " + type_name
            if not aligned:
                type_name = "unaligned " + type_name
            name_parts = (type_name, dtype, ndim, layout, usm_type)
            name = "%s(%s, %sd, %s, %s)" % name_parts

        if usm_type is None:
            usm_type = "device"

        self.usm_type = usm_type
        super().__init__(
            dtype,
            ndim,
            layout,
            readonly=readonly,
            name=name,
            addrspace=addrspace,
        )

    @property
    def key(self):
        return (*super().key, self.usm_type)

    @property
    def as_array(self):
        return self

    @property
    def box_type(self):
        return ndarray
