# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import random

from dpctl import SyclQueue
from numba import types
from numba.core import cgutils
from numba.extending import NativeValue, box, unbox

from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.runtime import context as dpexrt


class DpctlSyclQueue(types.Type):
    """A Numba type to represent a dpctl.SyclQueue PyObject.

    For now, a dpctl.SyclQueue is represented as a Numba opaque type that allows
    passing in and using a SyclQueue object as an opaque pointer type inside
    Numba.
    """

    def __init__(self, sycl_queue):
        self._sycl_queue = sycl_queue
        super(DpctlSyclQueue, self).__init__(name="DpctlSyclQueue")

    @property
    def sycl_queue(self):
        return self._sycl_queue

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
        return hash(self._sycl_queue)

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

    if c.context.enable_nrt:
        dpexrtCtx = dpexrt.DpexRTContext(c.context)
        errcode = dpexrtCtx.queuestruct_from_python(c.pyapi, obj, ptr)
    else:
        raise UnreachableError
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
