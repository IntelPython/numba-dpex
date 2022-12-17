# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor
from numba.core.typeconv import Conversion
from numba.core.types.npytypes import Array

"""A type class to represent dpctl.tensor.usm_ndarray type in Numba
"""


class USMNdArrayType(Array):
    """A type class to represent dpctl.tensor.usm_ndarray."""

    def __init__(
        self,
        dtype,
        ndim,
        layout,
        usm_type,
        device,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=None,
    ):
        if name is None:
            type_name = "usm_ndarray"
            if readonly:
                type_name = "readonly " + type_name
            if not aligned:
                type_name = "unaligned " + type_name
            name_parts = (type_name, dtype, ndim, layout, usm_type)
            name = "%s(%s, %sd, %s, %s)" % name_parts

        self.usm_type = usm_type
        self.device = device
        self.addrspace = addrspace
        super().__init__(
            dtype,
            ndim,
            layout,
            readonly=readonly,
            name=name,
            aligned=aligned,
        )

    def copy(
        self,
        dtype=None,
        ndim=None,
        layout=None,
        readonly=None,
        addrspace=None,
        device=None,
        usm_type=None,
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
        if device is None:
            device = self.device
        if usm_type is None:
            usm_type = self.usm_type
        return USMNdArrayType(
            dtype=dtype,
            ndim=ndim,
            layout=layout,
            usm_type=usm_type,
            device=device,
            readonly=readonly,
            aligned=self.aligned,
            addrspace=addrspace,
        )

    def unify(self, typingctx, other):
        """
        Unify this with the *other* USMNdArrayType.
        """
        # If other is array and the ndim matches
        if isinstance(other, USMNdArrayType) and other.ndim == self.ndim:
            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                if self.layout == other.layout:
                    layout = self.layout
                else:
                    layout = "A"
                readonly = not (self.mutable and other.mutable)
                aligned = self.aligned and other.aligned
                return USMNdArrayType(
                    dtype=self.dtype,
                    ndim=self.ndim,
                    layout=layout,
                    readonly=readonly,
                    aligned=aligned,
                )

    def can_convert_to(self, typingctx, other):
        """
        Convert this USMNdArrayType to the *other*.
        """
        if (
            isinstance(other, USMNdArrayType)
            and other.ndim == self.ndim
            and other.dtype == self.dtype
            and other.usm_type == self.usm_type
            and other.device == self.device
        ):
            if (
                other.layout in ("A", self.layout)
                and (self.mutable or not other.mutable)
                and (self.aligned or not other.aligned)
            ):
                return Conversion.safe

    @property
    def key(self):
        return (*super().key, self.addrspace, self.usm_type, self.device)

    @property
    def as_array(self):
        return self

    @property
    def box_type(self):
        return dpctl.tensor.usm_ndarray
