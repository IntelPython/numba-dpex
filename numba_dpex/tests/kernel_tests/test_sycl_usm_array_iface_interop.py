# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from dpctl import tensor as dpt

import numba_dpex as dpex
from numba_dpex.tests._helper import get_all_dtypes


class DuckUSMArray:
    """A Python class that defines a __sycl_usm_array_interface__ attribute."""

    def __init__(self, shape, dtype="d", host_buffer=None):
        _tensor = dpt.empty(shape, dtype=dtype, usm_type="shared")
        shmem = _tensor.usm_data
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


@dpex.kernel
def vecadd(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)


@pytest.fixture(params=dtypes)
def dtype(request):
    return request.param


def test_kernel_valid_usm_obj(dtype):
    """Test if a ``numba_dpex.kernel`` function accepts a DuckUSMArray argument.

    The ``DuckUSMArray`` uses ``dpctl.memory`` to allocate a Python object that
    defines a ``__sycl_usm_array_interface__`` attribute. We test if
    ``numba_dpex`` recognizes the ``DuckUSMArray`` as a valid USM-backed Python
    object and accepts it as a kernel argument.

    """
    N = 1024

    buffA = np.arange(0, N, dtype=dtype)
    buffB = np.arange(0, N, dtype=dtype)
    buffC = np.zeros(N, dtype=dtype)

    A = DuckUSMArray(shape=buffA.shape, dtype=dtype, host_buffer=buffA)
    B = DuckUSMArray(shape=buffB.shape, dtype=dtype, host_buffer=buffB)
    C = DuckUSMArray(shape=buffC.shape, dtype=dtype, host_buffer=buffC)

    try:
        dpex.call_kernel(vecadd, dpex.Range(N), A, B, C)
    except Exception:
        pytest.fail(
            "Could not pass Python object with sycl_usm_array_interface"
            + " to a kernel."
        )


def test_kernel_invalid_usm_obj(dtype):
    """Test if a ``numba_dpex.kernel`` function rejects a PseudoDuckUSMArray
    argument.

    The ``PseudoDuckUSMArray`` defines a fake attribute called
    __sycl_usm_array__interface__. We test if
    ``numba_dpex`` correctly recognizes and rejects the ``PseudoDuckUSMArray``.

    """
    N = 1024

    A = PseudoDuckUSMArray()
    B = PseudoDuckUSMArray()
    C = PseudoDuckUSMArray()

    with pytest.raises(Exception):
        dpex.call_kernel(vecadd, dpex.Range(N), A, B, C)
