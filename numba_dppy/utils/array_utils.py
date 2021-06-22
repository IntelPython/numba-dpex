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

SYCL_USM_ARRAY_INTERFACE_ATTR = "__sycl_usm_array_interface__"

supported_numpy_dtype = [np.int32, np.int64, np.uint32, np.int64, np.float32, np.float64]

def is_device_accessible_array(arr):
    """
    Determine if a given array is SYCL device accessible.

    Args:
        arr: array like object.

    Returns:
        bool: True if array is device accessible, False otherwise.
    """
    if hasattr(arr, SYCL_USM_ARRAY_INTERFACE_ATTR):
        return True
    elif hasattr(arr, "base"):
        if hasattr(arr.base, SYCL_USM_ARRAY_INTERFACE_ATTR):
            return True

    return False


def as_usm_backed_ndarray(shape, dtype, queue):
    """
    Allocate USM shared buffer and create a numpy.ndarray using that buffer.

    Args:
        shape (int, tuple of int): Shape of the resulting array.
        dtype (numpy.dtype): Numpy dtype (numpy.int32, numpy.float32, etc.).
        queue (dpctl.SyclQueue): SYCL queue used to create the buffer.

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
        raise TypeError("dtype has to be of type numpy.dtype. Got %s" % (type(dtype)))
    else:
        if dtype not in [np.dtype(typ) for typ in supported_numpy_dtype]:
            raise ValueError("dtype is not supprted. Supported dtypes are: %s" % (supported_numpy_dtype))
    if not isinstance(queue, dpctl.SyclQueue):
        raise TypeError(
            "queue has to be of dpctl.SyclQueue type. Got %s"
            % (type(queue))
        )

    size = np.prod(shape)

    usm_buf = dpctl_mem.MemoryUSMShared(size * dtype.itemsize, queue=queue)
    usm_ndarr = np.ndarray(shape, buffer=usm_buf, dtype=dtype)

    return usm_ndarr


def to_usm_backed_ndarray(ndarr, queue):
    """
    Create a np.ndarray using USM shared buffer and copy the data from
    provided array to the newly allocated buffer.

    Args:
        ndarr (numpy.ndarray): Array data will be copied from.
        queue (dpctl.SyclQueue): SYCL queue used to create the buffer.

    Returns:
        numpy.ndarray: Device accessible NumPy array with same data as ndarr.

    Raises:
        TypeError: if any argument is not of permitted type.
    """
    if not isinstance(ndarr, np.ndarray):
        raise TypeError("ndarr has to be of type numpy.ndarray. Got %s" % (type(dtype)))

    usm_ndarr = as_device_accessible_array(ndarr.shape, ndarr.dtype, queue)
    np.copyto(usm_ndarr, ndarr)

    return usm_ndarr
