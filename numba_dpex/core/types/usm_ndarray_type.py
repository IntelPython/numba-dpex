# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""A type class to represent dpctl.tensor.usm_ndarray type in Numba
"""

import dpctl
import dpctl.tensor
from numba.core.typeconv import Conversion
from numba.core.types.npytypes import Array

from numba_dpex.utils import address_space


class USMNdArray(Array):
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
        addrspace=address_space.GLOBAL,
    ):
        self.usm_type = usm_type
        self.addrspace = addrspace

        # Normalize the device filter string and get the fully qualified three
        # tuple (backend:device_type:device_num) filter string from dpctl.
        _d = dpctl.SyclDevice(device)
        self.device = _d.filter_string

        if name is None:
            type_name = "usm_ndarray"
            if readonly:
                type_name = "readonly " + type_name
            if not aligned:
                type_name = "unaligned " + type_name
            name_parts = (
                type_name,
                dtype,
                ndim,
                layout,
                self.addrspace,
                usm_type,
                self.device,
            )
            name = (
                "%s(dtype=%s, ndim=%s, layout=%s, address_space=%s, "
                "usm_type=%s, sycl_device=%s)" % name_parts
            )

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
        return USMNdArray(
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
        Unify this with the *other* USMNdArray.
        """
        # If other is array and the ndim, usm_type, address_space, and device
        # attributes match

        if (
            isinstance(other, USMNdArray)
            and other.ndim == self.ndim
            and self.device == other.device
            and self.addrspace == other.addrspace
            and self.usm_type == other.usm_type
        ):
            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                if self.layout == other.layout:
                    layout = self.layout
                else:
                    layout = "A"
                readonly = not (self.mutable and other.mutable)
                aligned = self.aligned and other.aligned
                return USMNdArray(
                    dtype=self.dtype,
                    ndim=self.ndim,
                    layout=layout,
                    readonly=readonly,
                    aligned=aligned,
                    usm_type=self.usm_type,
                    device=self.device,
                    addrspace=self.addrspace,
                )

    def can_convert_to(self, typingctx, other):
        """
        Convert this USMNdArray to the *other*.
        """
        if (
            isinstance(other, USMNdArray)
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
