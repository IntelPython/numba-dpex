import dpctl
import pytest

from numba_dpex.core.types import USMNdArray, dpctl_types


def test_usmndarray_negative_tests():
    default_device = dpctl.SyclDevice().filter_string

    usmarr1 = USMNdArray(1, device=None, queue=None)
    assert usmarr1.dtype.name == "float64"
    assert usmarr1.ndim == 1
    assert usmarr1.layout == "C"
    assert usmarr1.addrspace == 1
    assert usmarr1.usm_type == "device"

    assert usmarr1.queue.sycl_device == default_device

    usmarr2 = USMNdArray(1, device=default_device, queue=None)
    assert usmarr2.dtype.name == "float64"
    assert usmarr2.ndim == 1
    assert usmarr2.layout == "C"
    assert usmarr2.addrspace == 1
    assert usmarr2.usm_type == "device"
    assert usmarr2.queue.sycl_device == default_device

    queue = dpctl_types.DpctlSyclQueue(dpctl.SyclQueue())

    usmarr3 = USMNdArray(1, device=None, queue=queue)
    assert usmarr3.dtype.name == "float64"
    assert usmarr3.ndim == 1
    assert usmarr3.layout == "C"
    assert usmarr3.addrspace == 1
    assert usmarr3.usm_type == "device"

    with pytest.raises(TypeError):
        USMNdArray(1, device=default_device, queue=queue)

    with pytest.raises(TypeError):
        USMNdArray(1, queue=0)

    with pytest.raises(TypeError):
        USMNdArray(1, device=0)
