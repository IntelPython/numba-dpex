import dpctl.memory as dpmem
import numpy as np


class DuckUSMArray:
    """A Python class that defines a __sycl_usm_array_interface__ attribute."""

    def __init__(self, shape, dtype="d", host_buffer=None):
        nelems = np.prod(shape)
        bytes = nelems * np.dtype(dtype).itemsize
        shmem = dpmem.MemoryUSMShared(bytes)
        if isinstance(host_buffer, np.ndarray):
            shmem.copy_from_host(host_buffer.view(dtype="|u1"))
        self.arr = np.ndarray(shape, dtype=dtype, buffer=shmem)

    def __getitem__(self, indx):
        return self.arr[indx]

    def __setitem__(self, indx, val):
        self.arr.__setitem__(indx, val)

    @property
    def __sycl_usm_array_interface__(self):
        iface = self.arr.__array_interface__
        b = self.arr.base
        iface["syclobj"] = b.__sycl_usm_array_interface__["syclobj"]
        iface["version"] = 1
        return iface


class PseudoDuckUSMArray:
    """A Python class that defines an attributed called
    __sycl_usm_array_interface__, but is not actually backed by USM memory.

    """

    def __init__(self):
        pass

    @property
    def __sycl_usm_array_interface__(self):
        iface = {}
        iface["syclobj"] = None
        iface["version"] = 0
        return iface
