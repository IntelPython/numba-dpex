# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import random

from dpctl import SyclEvent, SyclQueue
from numba import types
from numba.core import cgutils
from numba.extending import NativeValue, box, unbox

from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.runtime import context as dpexrt


class DpctlSyclQueue(types.Type):
    """A Numba type to represent a dpctl.SyclQueue PyObject."""

    def __init__(self, sycl_queue):
        if not isinstance(sycl_queue, SyclQueue):
            raise TypeError("The argument sycl_queue is not of type SyclQueue.")

        # XXX: Storing the device filter string is a temporary workaround till
        # the compute follows data inference pass is fixed to use SyclQueue
        self._device = sycl_queue.sycl_device.filter_string
        self._device_has_aspect_atomic64 = (
            sycl_queue.sycl_device.has_aspect_atomic64
        )
        try:
            self._unique_id = hash(sycl_queue)
        except Exception:
            self._unique_id = self.rand_digit_str(16)
        super(DpctlSyclQueue, self).__init__(
            name=f"DpctlSyclQueue on {self._device}"
        )

    def rand_digit_str(self, n):
        return "".join(
            ["{}".format(random.randint(0, 9)) for num in range(0, n)]
        )

    @property
    def sycl_device(self):
        """Returns the SYCL oneAPI extension filter string associated with the
        queue.

        Returns:
            str: A SYCL oneAPI extension filter string
        """
        return self._device

    @property
    def device_has_aspect_atomic64(self):
        return self._device_has_aspect_atomic64

    @property
    def key(self):
        """Returns a Python object used as the key to cache an instance of
        DpctlSyclQueue.

        The key is constructed by hashing the actual dpctl.SyclQueue object
        encapsulated by an instance of DpctlSyclQueue. Doing so ensures, that
        different dpctl.SyclQueue instances are inferred as separate instances
        of the DpctlSyclQueue type.

        Returns:
            int: hash of the self._sycl_queue Python object.
        """
        return self._unique_id

    @property
    def box_type(self):
        return SyclQueue


@unbox(DpctlSyclQueue)
def unbox_sycl_queue(typ, obj, c):
    """
    Convert a SyclQueue object to a native structure.
    """

    qstruct = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    qptr = qstruct._getpointer()
    ptr = c.builder.bitcast(qptr, c.pyapi.voidptr)

    dpexrtCtx = dpexrt.DpexRTContext(c.context)
    errcode = dpexrtCtx.queuestruct_from_python(c.pyapi, obj, ptr)
    is_error = cgutils.is_not_null(c.builder, errcode)

    # Handle error
    with c.builder.if_then(is_error, likely=False):
        c.pyapi.err_set_string(
            "PyExc_TypeError",
            "can't unbox dpctl.SyclQueue from PyObject into a Numba "
            "native value. The object maybe of a different type",
        )

    return NativeValue(c.builder.load(qptr), is_error=is_error)


@box(DpctlSyclQueue)
def box_sycl_queue(typ, val, c):
    """Boxes a NativeValue representation of DpctlSyclQueue type into a
    dpctl.SyclQueue PyObject

    At this point numba-dpex does not support creating a dpctl.SyclQueue inside
    a dpjit decorated function. For this reason, boxing is only returns the
    original parent object stored in DpctlSyclQueue's data model.

    Args:
        typ: The representation of the DpnpNdArray type.
        val: A native representation of a Numba DpnpNdArray type object.
        c: The boxing context.

    Returns: A Pyobject for a dpnp.ndarray boxed from the Numba native value.
    """

    if c.context.enable_nrt:
        dpexrtCtx = dpexrt.DpexRTContext(c.context)
        queue = dpexrtCtx.queuestruct_to_python(c.pyapi, val)

        if not queue:
            c.pyapi.err_set_string(
                "PyExc_TypeError",
                "could not box native sycl queue into a dpctl.SyclQueue"
                " PyObject.",
            )
        return queue
    else:
        raise UnreachableError


class DpctlSyclEvent(types.Type):
    """A Numba type to represent a dpctl.SyclEvent PyObject."""

    def __init__(self):
        super(DpctlSyclEvent, self).__init__(name="DpctlSyclEvent")

    @property
    def box_type(self):
        return SyclEvent


@unbox(DpctlSyclEvent)
def unbox_sycl_event(typ, obj, c):
    """
    Convert a SyclEvent object to a native structure.
    """

    qstruct = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    qptr = qstruct._getpointer()
    ptr = c.builder.bitcast(qptr, c.pyapi.voidptr)

    dpexrtCtx = dpexrt.DpexRTContext(c.context)
    errcode = dpexrtCtx.eventstruct_from_python(c.pyapi, obj, ptr)
    is_error = cgutils.is_not_null(c.builder, errcode)

    # Handle error
    with c.builder.if_then(is_error, likely=False):
        c.pyapi.err_set_string(
            "PyExc_TypeError",
            "can't unbox dpctl.SyclEvent from PyObject into a Numba "
            "native value. The object maybe of a different type",
        )

    return NativeValue(c.builder.load(qptr), is_error=is_error)


@box(DpctlSyclEvent)
def box_sycl_event(typ, val, c):
    """Boxes a NativeValue representation of DpctlSyclEvent type into a
    dpctl.SyclEvent PyObject

    At this point numba-dpex does not support creating a dpctl.SyclEvent inside
    a dpjit decorated function. For this reason, boxing is only returns the
    original parent object stored in DpctlSyclEvent's data model.

    Args:
        typ: The representation of the dpctl.SyclEvent type.
        val: A native representation of a Numba DpctlSyclEvent type object.
        c: The boxing context.

    Returns: A Pyobject for a dpctl.SyclEvent boxed from the Numba native value.
    """

    if not c.context.enable_nrt:
        raise UnreachableError

    dpexrtCtx = dpexrt.DpexRTContext(c.context)
    event = dpexrtCtx.eventstruct_to_python(c.pyapi, val)

    if not event:
        c.pyapi.err_set_string(
            "PyExc_TypeError",
            "could not box native sycl queue into a dpctl.SyclEvent"
            " PyObject.",
        )
    return event
