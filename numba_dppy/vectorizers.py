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

"""Provide @vectorize(target="dppy") support."""

import warnings

import dpctl
import numpy as np
from numba.np.ufunc import deviceufunc

import numba_dppy as dppy
from numba_dppy.utils import (
    as_usm_obj,
    copy_to_numpy_from_usm_obj,
    has_usm_memory,
)

vectorizer_stager_source = """
def __vectorized_{name}({args}, __out__):
    __tid__ = __dppy__.get_global_id(0)
    if __tid__ < __out__.shape[0]:
        __out__[__tid__] = __core__({argitems})
"""


class DPPYVectorize(deviceufunc.DeviceVectorize):
    def _compile_core(self, sig):
        devfn = dppy.func(sig)(self.pyfunc)
        return devfn, devfn.cres.signature.return_type

    def _get_globals(self, corefn):
        glbl = self.pyfunc.__globals__.copy()
        glbl.update({"__dppy__": dppy, "__core__": corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        return dppy.kernel(sig)(fnobj)

    def build_ufunc(self):
        return DPPYUFuncDispatcher(self.kernelmap)

    @property
    def _kernel_template(self):
        return vectorizer_stager_source


class DPPYUFuncDispatcher(object):
    """
    Invoke the dppy ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels

    def __call__(self, *args, **kws):
        """
        Call the DPPY kernel launching mechanism
        Args:
            *args (np.ndarray): NumPy arrays
            **kws (optional):
                queue (dpctl._sycl_queue.SyclQueue): SYCL queue.
                out (np.ndarray): Output array.
        """
        return DPPYUFuncMechanism.call(self.functions, args, kws)

    def reduce(self, arg, queue=0):
        raise NotImplementedError


class DPPYUFuncMechanism(deviceufunc.UFuncMechanism):
    """
    Mechanism to process Input to a SYCL kernel and launch that kernel
    """

    @classmethod
    def call(cls, typemap, args, kws):
        """
        Perform the entire ufunc call mechanism.
        Args:
            typemap (dict): Signature mapped to kernel.
            args: Arguments to the @vectorize function.
            kws (optional): Optional keywords. Not supported.
        """
        # Handle keywords
        queue = dpctl.get_current_queue()
        out = kws.pop("out", None)

        if kws:
            warnings.warn("unrecognized keywords: %s" % ", ".join(kws))

        # Begin call resolution
        cr = cls(typemap, args)
        args = cr.get_arguments()
        resty, func = cr.get_function()

        outshape = args[0].shape

        # Adjust output value
        if out is not None and cr.is_device_array(out):
            out = cr.as_device_array(out)

        def attempt_ravel(a):
            if cr.SUPPORT_DEVICE_SLICING:
                raise NotImplementedError

            try:
                # Call the `.ravel()` method
                return a.ravel()
            except NotImplementedError:
                # If it is not a device array
                if not cr.is_device_array(a):
                    raise
                # For device array, retry ravel on the host by first
                # copying it back.
                else:
                    hostary = cr.to_host(a, queue).ravel()
                    return cr.to_device(hostary, queue)

        if args[0].ndim > 1:
            args = [attempt_ravel(a) for a in args]

        # Prepare argument on the device
        devarys = []
        any_device = True
        for a in args:
            if cr.is_device_array(a):
                devarys.append(a)
            else:
                dev_a = cr.to_device(a, queue=queue)
                devarys.append(dev_a)

        # Launch
        shape = args[0].shape
        if out is None:
            # No output is provided
            devout = cr.device_array(shape, resty, queue=queue)

            devarys.extend([devout])
            cr.launch(func, shape[0], queue, devarys)

            if any_device:
                # If any of the arguments are on device,
                # Keep output on the device
                return devout.reshape(outshape)
            else:
                # Otherwise, transfer output back to host
                # return devout.copy_to_host().reshape(outshape)
                raise ValueError("copy_to_host() is not yet supported")

        elif cr.is_device_array(out):
            # If output is provided and it is a device array,
            # Return device array
            if out.ndim > 1:
                out = attempt_ravel(out)
            devout = out
            devarys.extend([devout])
            cr.launch(func, shape[0], queue, devarys)
            return devout.reshape(outshape)

        else:
            # If output is provided and it is a host array,
            # Return host array
            assert out.shape == shape
            assert out.dtype == resty
            devout = cr.device_array(shape, resty, queue=queue)
            devarys.extend([devout])
            cr.launch(func, shape[0], queue, devarys)
            return devout.reshape(outshape)

    def as_device_array(self, obj):
        return obj

    def is_device_array(self, obj):
        ret = has_usm_memory(obj)
        return ret is not None

    def is_host_array(self, obj):
        return not self.is_device_array(obj)

    def to_device(self, hostary, queue):
        usm_mem = as_usm_obj(hostary, queue=queue, usm_type="shared")
        usm_backed_ndary = np.ndarray(
            hostary.shape, buffer=usm_mem, dtype=hostary.dtype
        )
        return usm_backed_ndary

    def to_host(self, devary, queue):
        hostary = np.empty(devary.shape, dtype=devary.dtype)
        devary_memview = memoryview(devary)
        devary_memview = devary_memview.cast("B")
        copy_to_numpy_from_usm_obj(devary_memview, hostary)

    def launch(self, func, count, queue, args):
        func[count, dppy.DEFAULT_LOCAL_SIZE](*args)

    def device_array(self, shape, dtype, queue):
        size = np.prod(shape)
        itemsize = dtype.itemsize
        usm_mem = dpctl.memory.MemoryUSMShared(size * itemsize, queue=queue)
        usm_backed_ndary = np.ndarray(shape, buffer=usm_mem, dtype=dtype)
        return usm_backed_ndary

    def broadcast_device(self, ary, shape):
        raise NotImplementedError("device broadcast_device NIY")
