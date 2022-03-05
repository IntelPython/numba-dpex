import dpctl
import numba_dpex as dppy
import numpy as np
import pytest

from .DuckUSMArray import DuckUSMArray, PseudoDuckUSMArray


@dppy.kernel
def vecadd(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


dtypes = [
    "i4",
    "i8",
    "f4",
    "f8",
]


@pytest.fixture(params=dtypes)
def dtype(request):
    return request.param


def test_dppy_kernel_valid_usm_obj(dtype):
    """Test if a ``numba_dpex.kernel`` function accepts a DuckUSMArray argument.

    The ``DuckUSMArray`` uses ``dpctl.memory`` to allocate a Python object that
    defines a __sycl_usm_array__interface__ attribute. We test if
    ``numba_dpex`` recognizes the ``DuckUSMArray`` as a valid USM-backed Python
    object and accepts it as a kernel argument.

    """
    N = 1024

    buffA = np.arange(0, N, dtype=dtype)
    A = DuckUSMArray(shape=buffA.shape, dtype=dtype, host_buffer=buffA)

    buffB = np.arange(0, N, dtype=dtype)
    B = DuckUSMArray(shape=buffB.shape, dtype=dtype, host_buffer=buffB)

    buffC = np.zeros(N, dtype=dtype)
    C = DuckUSMArray(shape=buffC.shape, dtype=dtype, host_buffer=buffC)

    try:
        with dpctl.device_context(dpctl.select_default_device()):
            vecadd[N, dppy.DEFAULT_LOCAL_SIZE](A, B, C)
    except Exception:
        pytest.fail(
            "Could not pass Python object with sycl_usm_array_interface"
            + " to a kernel."
        )


def test_dppy_kernel_invalid_usm_obj(dtype):
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
        with dpctl.device_context(dpctl.select_default_device()):
            vecadd[N, dppy.DEFAULT_LOCAL_SIZE](A, B, C)
