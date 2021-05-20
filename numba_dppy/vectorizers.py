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

from numba.np.ufunc import deviceufunc
import numba_dppy as dppy
from numba_dppy.dppy_offload_dispatcher import DppyOffloadDispatcher
from numba_dppy.compiler import (is_device_array as dppy_is_device_array,
                                 device_array as dppy_device_array,
                                 to_device as dppy_to_device)
from numba_dppy.descriptor import dppy_target
import dpctl

vectorizer_stager_source = '''
def __vectorized_{name}({args}, __out__):
    __tid__ = __dppy__.get_global_id(0)
    if __tid__ < __out__.shape[0]:
        __out__[__tid__] = __core__({argitems})
'''


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
    Invoke the HSA ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels
        #self.functions = tuple([self.typingctx.resolve_argument_type(a) for a in args])

    def __call__(self, *args, **kws):
        """
        *args: numpy arrays
        **kws:
            stream -- hsa stream; when defined, asynchronous mode is used.
            out    -- output array. Can be a numpy array or DeviceArrayBase
                      depending on the input arguments.  Type must match
                      the input arguments.
        """
        return DPPYUFuncMechanism.call(self.functions, args, kws)

    def reduce(self, arg, stream=0):
        raise NotImplementedError


class DPPYUFuncMechanism(deviceufunc.UFuncMechanism):
    """
    Provide OpenCL specialization
    """
    @classmethod
    def call(cls, typemap, args, kws):
        """Perform the entire ufunc call mechanism.
        """
        # Handle keywords
        stream = dpctl.get_current_queue()
        out = kws.pop('out', None)

        if kws:
            warnings.warn("unrecognized keywords: %s" % ', '.join(kws))

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
                    hostary = cr.to_host(a, stream).ravel()
                    return cr.to_device(hostary, stream)

        if args[0].ndim > 1:
            args = [attempt_ravel(a) for a in args]

        # Prepare argument on the device
        devarys = []
        any_device = True
        for a in args:
            if cr.is_device_array(a):
                devarys.append(a)
            else:
                dev_a = cr.to_device(a, stream=stream)
                devarys.append(dev_a)

        # Launch
        shape = args[0].shape
        if out is None:
            # No output is provided
            devout = cr.device_array(shape, resty, stream=stream)

            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)

            if any_device:
                # If any of the arguments are on device,
                # Keep output on the device
                return devout.reshape(outshape)
            else:
                # Otherwise, transfer output back to host
                #return devout.copy_to_host().reshape(outshape)
                raise ValueError("copy_to_host() is not yet supported")

        elif cr.is_device_array(out):
            # If output is provided and it is a device array,
            # Return device array
            if out.ndim > 1:
                out = attempt_ravel(out)
            devout = out
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.reshape(outshape)

        else:
            # If output is provided and it is a host array,
            # Return host array
            assert out.shape == shape
            assert out.dtype == resty
            devout = cr.device_array(shape, resty, stream=stream)
            devarys.extend([devout])
            cr.launch(func, shape[0], stream, devarys)
            return devout.reshape(outshape)

    def is_device_array(self, obj):
        return dppy_is_device_array(obj)

    def is_host_array(self, obj):
        return not dppy_is_device_array(obj)

    def to_device(self, hostary, stream):
        return dppy_to_device(hostary, stream)

    def to_host(self, devary, stream):
        raise NotImplementedError('device to_host NIY')

    def launch(self, func, count, stream, args):
        func[count, dppy.DEFAULT_LOCAL_SIZE](*args)

    def device_array(self, shape, dtype, stream):
        return dppy_device_array(shape, dtype, stream)

    def broadcast_device(self, ary, shape):
        raise NotImplementedError('device broadcast_device NIY')
