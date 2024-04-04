# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""A type class to represent dpctl.tensor.usm_ndarray type in Numba
"""

import dpctl
import dpctl.tensor
from numba import types
from numba.core.typeconv import Conversion
from numba.core.types.npytypes import Array
from numba.np.numpy_support import from_dtype

from numba_dpex.core.types.dpctl_types import DpctlSyclQueue
from numba_dpex.kernel_api.memory_enums import AddressSpace as address_space


class USMNdArray(Array):
    """A type class to represent dpctl.tensor.usm_ndarray."""

    def __init__(
        self,
        ndim,
        layout="C",
        dtype=None,
        usm_type="device",
        device=None,
        queue=None,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=address_space.GLOBAL.value,
    ):
        if (
            queue is not None
            and not (
                isinstance(queue, types.misc.Omitted)
                or isinstance(queue, types.misc.NoneType)
            )
            and device is not None
        ):
            raise TypeError(
                "numba_dpex.core.types.usm_ndarray_type.USMNdArray.__init__(): "
                "`device` and `sycl_queue` are exclusive keywords, "
                "i.e. use one or other."
            )

        if queue is not None and not (
            isinstance(queue, types.misc.Omitted)
            or isinstance(queue, types.misc.NoneType)
        ):
            if not isinstance(queue, DpctlSyclQueue):
                raise TypeError(
                    "The queue keyword arg should be either DpctlSyclQueue or "
                    "NoneType. Found type(queue) = " + str(type(queue))
                )
            self.queue = queue
        else:
            if device is None:
                sycl_device = dpctl.SyclDevice()
            else:
                if not isinstance(device, str):
                    raise TypeError(
                        "The device keyword arg should be a str object "
                        "specifying a SYCL filter selector."
                    )
                sycl_device = dpctl.SyclDevice(device)

            sycl_queue = dpctl._sycl_queue_manager.get_device_cached_queue(
                sycl_device
            )
            self.queue = DpctlSyclQueue(sycl_queue=sycl_queue)

        self.device = self.queue.sycl_device
        self.usm_type = usm_type
        self.addrspace = addrspace

        if not dtype:
            dummy_tensor = dpctl.tensor.empty(
                1, order=layout, usm_type=usm_type, device=self.device
            )
            # convert dpnp type to numba/numpy type
            _dtype = dummy_tensor.dtype
            self.dtype = from_dtype(_dtype)
        else:
            self.dtype = dtype

        if name is None:
            type_name = "USMNdArray"
            if readonly:
                type_name = "readonly " + type_name
            if not aligned:
                type_name = "unaligned " + type_name
            name_parts = (
                type_name,
                self.dtype,
                ndim,
                layout,
                self.addrspace,
                usm_type,
                self.device,
                self.queue,
            )
            name = (
                "%s(dtype=%s, ndim=%s, layout=%s, address_space=%s, "
                "usm_type=%s, device=%s, sycl_queue=%s)" % name_parts
            )

        super().__init__(
            self.dtype,
            ndim,
            layout,
            readonly=readonly,
            name=name,
            aligned=aligned,
        )

    def __repr__(self):
        return self.name

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
        return type(self)(
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
                return type(self)(
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
        return (
            *super().key,
            self.addrspace,
            self.usm_type,
            self.device,
            self.queue,
        )

    @property
    def as_array(self):
        return self

    @property
    def box_type(self):
        return dpctl.tensor.usm_ndarray

    @property
    def mangling_args(self):
        """Returns a list of parameters used to create a mangled name for a
        USMNdArray type.
        """
        filter_str_splits = self.device.split(":")
        args = [
            self.dtype,
            self.ndim,
            self.layout,
            filter_str_splits[0] + "_" + filter_str_splits[1],
        ]
        return self.__class__.__name__, args
