# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dpctl import SyclQueue
from numba import types
from numba.extending import NativeValue, box, type_callable, unbox


class DpctlSyclQueue(types.Type):
    """A Numba type to represent a dpctl.SyclQueue PyObject.

    For now, a dpctl.SyclQueue is represented as a Numba opaque type that allows
    passing in and using a SyclQueue object as an opaque pointer type inside
    Numba.
    """

    def __init__(self):
        super().__init__(name="DpctlSyclQueueType")


sycl_queue_ty = DpctlSyclQueue()


@type_callable(SyclQueue)
def type_interval(context):
    def typer():
        return sycl_queue_ty

    return typer


@unbox(DpctlSyclQueue)
def unbox_sycl_queue(typ, obj, c):
    return NativeValue(obj)


@box(DpctlSyclQueue)
def box_pyobject(typ, val, c):
    return val
