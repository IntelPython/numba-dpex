# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba.core.types.npytypes import Array


class Array(Array):
    """
    An array type for use inside our compiler pipeline.
    """

    def __init__(
        self,
        dtype,
        ndim,
        layout,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=None,
    ):
        self.addrspace = addrspace
        super(Array, self).__init__(
            dtype,
            ndim,
            layout,
            readonly=readonly,
            name=name,
            aligned=aligned,
        )

    def copy(
        self, dtype=None, ndim=None, layout=None, readonly=None, addrspace=None
    ):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        if addrspace is None:
            addrspace = self.addrspace
        return Array(
            dtype=dtype,
            ndim=ndim,
            layout=layout,
            readonly=readonly,
            aligned=self.aligned,
            addrspace=addrspace,
        )

    @property
    def key(self):
        return (*super().key, self.addrspace)

    @property
    def box_type(self):
        return np.ndarray

    def is_precise(self):
        return self.dtype.is_precise()
