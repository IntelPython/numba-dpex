from collections import namedtuple

import dpctl.tensor as dpt
import dpnp
import numpy as np
from dpctl.tensor._ctors import _coerce_and_infer_dt
from llvmlite import ir as llvmir
from numba import errors, types
from numba.core import cgutils
from numba.core.types.scalars import Complex, Float, Integer
from numba.extending import intrinsic, overload

import numba_dpex.utils as utils
from numba_dpex.core.runtime import context as dpexrt
from numba_dpex.core.types import DpnpNdArray
from numba_dpex.dpnp_iface._intrinsic import _get_queue_ref
from numba_dpex.dpnp_iface.arrayobj import (
    _parse_device_filter_string,
    _parse_usm_type,
)

_QueueRefPayload = namedtuple(
    "QueueRefPayload", ["queue_ref", "py_dpctl_sycl_queue_addr", "pyapi"]
)


def _parse_dtype(a):
    if isinstance(a.dtype, Complex):
        v_type = a.dtype
        w_type = dpnp.float64 if a.dtype.bitwidth == 128 else dpnp.float32
    elif isinstance(a.dtype, Float):
        v_type = w_type = a.dtype
    elif isinstance(a.dtype, Integer):
        v_type = w_type = (
            dpnp.float32 if a.dtype.bitwidth == 32 else dpnp.float64
        )
    # elif a.queue.sycl_device.has_aspect_fp64:
    #     v_type = w_type = dpnp.float64
    else:
        v_type = w_type = dpnp.float64
    return (v_type, w_type)


@overload(dpnp.arange, prefer_literal=True)
def ol_dpnp_arange(
    start,
    stop=None,
    step=1,
    dtype=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    print("start =", start, ", type(start) =", type(start))
    print("stop =", stop, ", type(stop) =", type(stop))
    print("step =", step, ", type(step) =", type(step))
    print("dtype =", dtype, ", type(dtype) =", type(dtype))
    print("device =", device, ", type(device) =", type(device))
    print("usm_type =", usm_type, ", type(usm_type) =", type(usm_type))
    print("sycl_queue =", sycl_queue, ", type(sycl_queue) =", type(sycl_queue))
    print("---")

    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    _dtype = _parse_dtype(dtype) if dtype is not None else type(start)
    _device = _parse_device_filter_string(device) if device else None
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"

    ret_ty = DpnpNdArray(
        ndim=1,
        layout="C",
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=sycl_queue,
    )

    print("start =", start, ", type(start) =", type(start))
    print("stop =", stop, ", type(stop) =", type(stop))
    print("step =", step, ", type(step) =", type(step))
    print("_dtype =", _dtype, ", type(_dtype) =", type(_dtype))
    print("_device =", _device, ", type(_device) =", type(_device))
    print("_usm_type =", _usm_type, ", type(_usm_type) =", type(_usm_type))
    print("sycl_queue =", sycl_queue, ", type(sycl_queue) =", type(sycl_queue))
    print("***")

    if ret_ty:

        def impl(
            start,
            stop=None,
            step=1,
            dtype=None,
            device=None,
            usm_type="device",
            sycl_queue=None,
        ):
            print("start =", start, ", type(start) =", type(start))
            print("stop =", stop, ", type(stop) =", type(stop))
            print("step =", step, ", type(step) =", type(step))
            print(
                "dtype =", dtype
            )  # , ", type(dtype) =", type(dtype) if dtype is not None else "Null")
            print(
                "device =", device
            )  # , ", type(device) =", type(device) if device is not None else "Null")
            print(
                "usm_type =", usm_type
            )  # , ", type(usm_type) =", type(usm_type) if usm_type is not None else "Null")
            print(
                "sycl_queue =", sycl_queue
            )  # , ", type(sycl_queue) =", type(sycl_queue) if sycl_queue is not None else "Null")
            print("###")

            v = dpnp.empty(10)
            return v

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.arange({start}, {stop}, {step}, ...)."
        )
