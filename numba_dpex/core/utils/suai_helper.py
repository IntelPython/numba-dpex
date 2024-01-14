# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging

import dpctl
import dpctl.memory as dpctl_mem
import numpy as np


class SyclUSMArrayInterface:
    """Stores as attributes the information extracted from a
    __sycl_usm_array_interface__ dictionary as defined by dpctl.memory.Memory*
    classes.
    """

    def __init__(
        self,
        data,
        writable,
        size,
        shape,
        dimensions,
        itemsize,
        strides,
        dtype,
        usm_type,
        device,
        queue,
    ):
        self._data = data
        self._data_writeable = writable
        self._size = size
        self._shape = shape
        self._dimensions = dimensions
        self._itemsize = itemsize
        self._strides = strides
        self._dtype = dtype
        self._usm_type = usm_type
        self._device = device
        self._queue = queue

    @property
    def data(self):
        return self._data

    @property
    def is_writable(self):
        return self._data_writeable

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def itemsize(self):
        return self._itemsize

    @property
    def strides(self):
        return self._strides

    @property
    def dtype(self):
        return self._dtype

    @property
    def usm_type(self):
        return self._usm_type

    @property
    def device(self):
        return self._device

    @property
    def queue(self):
        return self._queue


def get_info_from_suai(obj):
    """
    Extracts the metadata of an object of type UsmNdArray using the objects
    __sycl_usm_array_interface__ (SUAI) attribute.

    The ``dpctl.memory.as_usm_memory`` function converts the array-like
    object into a dpctl.memory.USMMemory object. Using the ``as_usm_memory``
    is an implicit way to verify if the array-like object is a legal
    SYCL USM memory back Python object that can be passed to a dpex kernel.

    Args:
        obj: array-like object with a SUAI attribute.

    Returns:
        A SyclUSMArrayInterface object

    """

    # dpctl.as_usm_memory validated if an array-like object, obj, has a well
    # defined __sycl_usm_array_interface__ dictionary and converts it into a
    # dpctl.memory.Memory* object.
    try:
        usm_mem = dpctl_mem.as_usm_memory(obj)
    except Exception as e:
        logging.exception(
            "Array like object with __sycl_usm_array_interface__ could not be "
            "converted to a dpctl.memory.Memory* object."
        )
        raise e

    # The data attribute of __sycl_usm_array_interface__ is a 2-tuple.
    # The first element is the data pointer and the second a boolean
    # value indicating if the data is writable.
    is_writable = usm_mem.__sycl_usm_array_interface__["data"][1]

    shape = obj.__sycl_usm_array_interface__["shape"]
    total_size = np.prod(shape)
    ndim = len(shape)
    dtype = np.dtype(obj.__sycl_usm_array_interface__["typestr"])
    itemsize = dtype.itemsize

    strides = obj.__sycl_usm_array_interface__["strides"]
    if strides is None:
        strides = [1] * ndim
        for i in reversed(range(1, ndim)):
            strides[i - 1] = strides[i] * shape[i]
        strides = tuple(strides)

    syclobj = usm_mem.sycl_queue
    device = syclobj.sycl_device.filter_string
    usm_type = usm_mem.get_usm_type()

    suai_info = SyclUSMArrayInterface(
        data=usm_mem,
        writable=is_writable,
        size=total_size,
        usm_type=usm_type,
        device=device,
        queue=syclobj,
        shape=shape,
        dimensions=ndim,
        itemsize=itemsize,
        strides=strides,
        dtype=dtype,
    )

    return suai_info
