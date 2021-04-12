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
import dpctl.memory as dpctl_mem
import dpctl
from numba.core import types
from numba.np import numpy_support


class DPPYDeviceArray(object):
    """Device Array type for dppy"""

    def __init__(self, shape, strides, dtype, usm_memory=None, queue=None):
        """
        Args
        ----

        shape
            array shape.
        strides
            array strides.
        dtype
            data type as numpy.dtype.
        usm_memory
            user provided device memory for the ndarray data buffer
        """

        # dpctl.memory.MemoryUSMShared does not cache the queue. Without
        # a reference to the Sycl queue in the usm_memory we need to
        # store the reference in this class. Until dpctl.memory.MemoryUSMShared
        # provides a reference to the queue we can not allow any usm_memory
        # being passed to this class.
        if usm_memory is not None:
            raise ValueError("Pre-allocated usm_memory is not supported")

        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(strides, int):
            strides = (strides,)
        self.ndim = len(shape)
        if len(strides) != self.ndim:
            raise ValueError("strides not match ndim")

        if queue is None:
            queue = dpctl.get_current_queue()

        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.dtype = np.dtype(dtype)
        self.size = int(np.prod(self.shape))
        self.itemsize = dtype.itemsize
        self.alloc_size = self.size * self.itemsize
        self.queue = queue

        if self.size > 0:
            if usm_memory is None:
                self.base = dpctl_mem.MemoryUSMShared(
                    self.size * self.itemsize, queue=queue
                )
            else:
                # we should never reach here, refer to comment above
                self.base = usm_memory
        else:
            self.base = None

        self.hostary = np.frombuffer(self.base, dtype=self.dtype).reshape(self.shape)

    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
        dtype = numpy_support.from_dtype(self.dtype)
        return types.Array(dtype, self.ndim, "A")

    def copy_to_device(self, ary):
        """Copy `ary` to `self`.

        Perform a a host-to-device transfer.
        """
        if ary.size == 0:
            # Nothing to do
            return

        # We only support copying from NumPy array
        assert isinstance(ary, np.ndarray)

        # copy to usm_memory
        self.base.copy_from_host(ary.tobytes())

    def copy_to_host(self, ary=None):
        """Copy ``self`` to ``ary`` or create a new Numpy ndarray
        if ``ary`` is ``None``.

        The transfer is synchronous: the function returns after the copy
        is finished.

        Always returns the host array.
        """
        # a location for the data exists as `hostary`
        assert self.alloc_size >= 0, "Negative memory size"

        if ary is None:  # destination does not exist
            if self.alloc_size != 0:
                ary = np.empty_like(self.hostary)
        else:  # destination does exist, it's `ary`, check it
            if ary.dtype != self.dtype:
                raise TypeError("incompatible dtype")

            if ary.shape != self.shape:
                scalshapes = (), (1,)
                if not (ary.shape in scalshapes and self.shape in scalshapes):
                    raise TypeError(
                        "incompatible shape; device %s; host %s"
                        % (self.shape, ary.shape)
                    )
            if ary.strides != self.strides:
                scalstrides = (), (self.dtype.itemsize,)
                if not (ary.strides in scalstrides and self.strides in scalstrides):
                    raise TypeError(
                        "incompatible strides; device %s; host %s"
                        % (self.strides, ary.strides)
                    )

        np.copyto(ary, self.hostary)
        return ary


def to_device(ary):
    """Convenience function to create a DPPYDeviceArray from a np.ndarray
    and copy data from ary to the created DPPYDeviceArray.

    Args
    ----
    ary
        np.ndarray.
    """
    if ary is None or not isinstance(ary, np.ndarray):
        raise ValueError("ary has to be a valid np.ndarray")

    da = DPPYDeviceArray(ary.shape, ary.strides, ary.dtype)
    da.copy_to_device(ary)

    return da
