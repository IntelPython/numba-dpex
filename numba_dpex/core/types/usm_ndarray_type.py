# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""A type class to represent dpctl.tensor.usm_ndarray type in Numba
"""

import dpctl
import dpctl.tensor
from numba.core.typeconv import Conversion
from numba.core.typeinfer import CallConstraint
from numba.core.types.npytypes import Array
from numba.np.numpy_support import from_dtype

from numba_dpex.utils import address_space


class USMNdArray(Array):
    """A type class to represent dpctl.tensor.usm_ndarray."""

    def __init__(
        self,
        ndim,
        layout="C",
        dtype=None,
        usm_type="device",
        device="unknown",
        queue=None,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=address_space.GLOBAL,
    ):
        self.usm_type = usm_type
        self.addrspace = addrspace

        self.readonly = readonly
        self.aligned = aligned
        self.layout = layout

        self.ndim = ndim

        if queue is not None and device != "unknown":
            if not isinstance(device, str):
                raise TypeError(
                    "The device keyword arg should be a str object specifying "
                    "a SYCL filter selector"
                )
            if not isinstance(queue, dpctl.SyclQueue):
                raise TypeError(
                    "The queue keyword arg should be a dpctl.SyclQueue object"
                )
            d1 = queue.sycl_device
            d2 = dpctl.SyclDevice(device)
            if d1 != d2:
                raise TypeError(
                    "The queue keyword arg and the device keyword arg specify "
                    "different SYCL devices"
                )
            self.queue = queue
            self._device = device
        elif queue is None and device != "unknown":
            if not isinstance(device, str):
                raise TypeError(
                    "The device keyword arg should be a str object specifying "
                    "a SYCL filter selector"
                )
            self.queue = dpctl.SyclQueue(device)
            self._device = device
        elif queue is not None and device == "unknown":
            if not isinstance(queue, dpctl.SyclQueue):
                raise TypeError(
                    "The queue keyword arg should be a dpctl.SyclQueue object"
                )
            self._device = self.queue.sycl_device.filter_string
            self.queue = queue
        else:
            self.queue = dpctl.SyclQueue()
            self._device = self.queue.sycl_device.filter_string

        if not dtype:
            dummy_tensor = dpctl.tensor.empty(
                sh=1, order=layout, usm_type=usm_type, sycl_queue=self.queue
            )
            # convert dpnp type to numba/numpy type
            _dtype = dummy_tensor.dtype
            self.dtype = from_dtype(_dtype)
        else:
            self.dtype = dtype

        if name is None:
            self.name = self._construct_name()
        else:
            self.name = name

        super().__init__(
            self.dtype,
            ndim,
            layout,
            readonly=readonly,
            name=self.name,
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
            device = self._device
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
            and self._device == other.device
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
                    device=self._device,
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
            and other.device == self._device
        ):
            if (
                other.layout in ("A", self.layout)
                and (self.mutable or not other.mutable)
                and (self.aligned or not other.aligned)
            ):
                return Conversion.safe

    @property
    def key(self):
        return (*super().key, self.addrspace, self.usm_type, self._device)

    @property
    def as_array(self):
        return self

    @property
    def box_type(self):
        return dpctl.tensor.usm_ndarray

    def _construct_name(self):
        type_name = "usm_ndarray"
        if self.readonly:
            type_name = "readonly " + type_name
        if not self.aligned:
            type_name = "unaligned " + type_name
        name_parts = (
            type_name,
            self.dtype,
            self.ndim,
            self.layout,
            self.addrspace,
            self.usm_type,
            self._device,
            self.queue,
        )
        rt_name = (
            "%s(dtype=%s, ndim=%s, layout=%s, address_space=%s, "
            "usm_type=%s, device=%s, sycl_device=%s)" % name_parts
        )
        return rt_name

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, input):
        self._device = input
        self.queue = dpctl.SyclQueue(self._device)
        self.name = self._construct_name()
