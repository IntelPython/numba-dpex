import dpctl
from numba.core.types.scalars import Float

from numba_dpex.core.types import USMNdArray


def test_init():
    usma = USMNdArray(1, device=None, queue=None)
    assert usma.dtype.name == "float64"
    assert usma.ndim == 1
    assert usma.layout == "C"
    assert usma.addrspace == 1
    assert usma.usm_type == "device"
    assert (
        str(usma.queue.sycl_device.device_type) == "device_type.cpu"
        or str(usma.queue.sycl_device.device_type) == "device_type.gpu"
    )

    device = dpctl.SyclDevice().filter_string

    usma = USMNdArray(1, device=device, queue=None)
    assert usma.dtype.name == "float64"
    assert usma.ndim == 1
    assert usma.layout == "C"
    assert usma.addrspace == 1
    assert usma.usm_type == "device"
    assert (
        str(usma.queue.sycl_device.device_type) == "device_type.cpu"
        or str(usma.queue.sycl_device.device_type) == "device_type.gpu"
    )

    # usma = USMNdArray(1, device="gpu", queue=None)
    # assert usma.dtype.name == "int64"
    # assert usma.ndim == 1
    # assert usma.layout == "C"
    # assert usma.addrspace == 1
    # assert usma.usm_type == "device"
    # assert str(usma.queue.sycl_device.device_type) == "device_type.gpu"

    queue = dpctl.SyclQueue()
    usma = USMNdArray(1, device=None, queue=queue)
    assert usma.dtype.name == "float64"
    assert usma.ndim == 1
    assert usma.layout == "C"
    assert usma.addrspace == 1
    assert usma.usm_type == "device"
    assert usma.queue.addressof_ref() > 0

    try:
        usma = USMNdArray(1, device=device, queue=queue)
    except Exception as e:
        assert "exclusive keywords" in str(e)

    try:
        usma = USMNdArray(1, queue=0)
    except Exception as e:
        assert "queue keyword arg" in str(e)

    try:
        usma = USMNdArray(1, device=0)
    except Exception as e:
        assert "SYCL filter selector" in str(e)
