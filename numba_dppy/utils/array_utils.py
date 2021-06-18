# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import dpctl
import dpctl.memory as dpctl_mem

DEVICE_ARRAY_ATTR = "__sycl_usm_array_interface__"


def _is_device_accessible_array(arr):
    """
    Function to determine if a given array is SYCL device accessible.

    Args:
        arr: array_like object.

    Retruns:
        bool: True if array is device accessible, False otherwise.
    """
    if hasattr(arr, DEVICE_ARRAY_ATTR):
        return True
    elif hasattr(arr, "base"):
        if hasattr(arr.base, DEVICE_ARRAY_ATTR):
            return True

    return False


def _as_device_accessible_array(shape, dtype, queue):
    """
    Allocate USM shared buffer and create a np.ndarray
    using that buffer.

    Args:
        shape (int, tuple of int): Shape of the resulting array.
        dtype (np.dtype): Numpy dtype (np.int32, np.float32, etc.).
        queue (dpctl._sycl_queue.SyclQueue): SYCL queue used to create the buffer.

    Returns:
        np.ndarray: NumPy array created using a USM shared buffer.

    Raises:
        TypeError: if any argument is not of permitted type.
    """
    if not (isinstance(shape, int) or isinstance(shape, tuple)):
        raise TypeError(
            "Shape has to be a integer or tuple of integers. Got %s" % (type(shape))
        )
    if not isinstance(dtype, np.dtype):
        raise TypeError("dtype has to be of type np.dtype. Got %s" % (type(dtype)))
    if not isinstance(queue, dpctl._sycl_queue.SyclQueue):
        raise TypeError(
            "queue has to be of dpctl._sycl_queue.SyclQueue type. Got %s"
            % (type(queue))
        )

    size = np.prod(shape)

    usm_buf = dpctl_mem.MemoryUSMShared(size * dtype.itemsize, queue=queue)
    usm_ndarr = np.ndarray(shape, buffer=usm_buf, dtype=dtype)

    return usm_ndarr


def _to_device_accessible_array(hostary, queue):
    """
    Create a np.ndarray using USM shared buffer and copy the data from
    provided array to the newly allocated buffer.

    Args:
        hostary (np.ndarray): Array data will be copied from.
        queue (dpctl._sycl_queue.SyclQueue): SYCL queue used to create the buffer.

    Returns:
        np.ndarray: Device accessible NumPy array with same data as hostary.

    Raises:
        TypeError: if any argument is not of permitted type.
    """
    if not isinstance(hostary, np.ndarray):
        raise TypeError("hostary has to be of type np.ndarray. Got %s" % (type(dtype)))

    usm_ndarr = _as_device_accessible_array(hostary.shape, hostary.dtype, queue)
    np.copyto(usm_ndarr, hostary)

    return usm_ndarr
