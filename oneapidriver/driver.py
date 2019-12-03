from __future__ import absolute_import, division, print_function

from . import _numba_oneapi_pybindings
from cffi import FFI
from numpy import ndarray

ffi = FFI()

# Exception classes ######################################################
class OneapiGlueDriverError(Exception):
    """A problem with the Numba-OneApi-Glue Python driver code
    """
    pass


class DeviceNotFoundError(Exception):
    """The requested type of device is not available
    """
    pass


class UnsupportedTypeError(Exception):
    """When expecting either a DeviceArray or numpy.ndarray object
    """
    pass


def _raise_driver_error(fname, errcode):
    e = OneapiGlueDriverError("NUMBA_ONEAPI_FAILURE encountered")
    e.fname = fname
    e.code = errcode
    raise e


def _raise_device_not_found_error(fname):
    e = DeviceNotFoundError("This type of device not available on the system")
    e.fname = fname
    raise e


def _raise_unsupported_type_error(fname):
    e = UnsupportedTypeError("Type needs to be DeviceArray or a numpy.ndarray")
    e.fname = fname
    raise e


############################### DeviceArray class ########################
class DeviceArray:

    _buffObj = None
    _ndarray = None
    _buffSize = None

    def __init__(self, context_ptr, arr):

        if not isinstance(arr, ndarray):
            _raise_unsupported_type_error("DeviceArray constructor")

        # create a numba_oneapi_buffer_t object
        self._buffObj = _numba_oneapi_pybindings.ffi.new("buffer_t *")
        self._ndarray = arr
        self._buffSize = arr.itemsize * arr.size
        retval = (_numba_oneapi_pybindings
                  .lib
                  .create_numba_oneapi_rw_mem_buffer(context_ptr,
                                                     self._buffObj,
                                                     self._buffSize))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("create_numba_oneapi_runtime", -1)

    def __del__(self):
        retval = (_numba_oneapi_pybindings
                  .lib
                  .destroy_numba_oneapi_rw_mem_buffer(self._buffObj))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("create_numba_oneapi_runtime", -1)

    def get_buffer_obj(self):
        return self._buffObj

    def get_buffer_size(self):
        return self._buffSize

    def get_data_ptr(self):
        return ffi.cast("void*", self._ndarray.ctypes.data)

############################## Device class ##############################
class Device():

    _device_ptr = None
    _context_ptr = None
    _queue_ptr = None

    def __init__(self, device_ptr, context_ptr, queue_ptr):
        self._device_ptr = device_ptr
        self._context_ptr = context_ptr
        self._queue_ptr = queue_ptr
        pass

    def __del__(self):
        pass

    def retain_context(self):
        print('return first_cpu_conext.context after calling clRetinContext')
        retval = (_numba_oneapi_pybindings
                  .lib
                  .retain_numba_oneapi_context(self._context))
        if(retval == -1):
            _raise_driver_error("retain_numba_oneapi_context", -1)

        return (self.__cpu_context)

    def release_context(self):
        retval = (_numba_oneapi_pybindings
                  .lib
                  .release_numba_oneapi_context(self._context))
        if retval == -1:
            _raise_driver_error("release_numba_oneapi_context", -1)

    def copy_array_to_device(self, array):
        if isinstance(array, DeviceArray):
            retval = (_numba_oneapi_pybindings
                      .lib
                      .write_numba_oneapi_mem_buffer_to_device(
                          self._queue_ptr,
                          array.get_buffer_obj()[0],
                          True,
                          0,
                          array.get_buffer_size(),
                          array.get_data_ptr()))
            if retval == -1:
                print("Error Code  : ", retval)
                _raise_driver_error("write_numba_oneapi_mem_buffer_to_device",
                                    -1)
            return array
        elif isinstance(array, ndarray):
            dArr = DeviceArray(self._context_ptr, array)
            retval = (_numba_oneapi_pybindings
                      .lib
                      .write_numba_oneapi_mem_buffer_to_device(
                          self._queue_ptr,
                          dArr.get_buffer_obj()[0],
                          True,
                          0,
                          dArr.get_buffer_size(),
                          dArr.get_data_ptr()))
            if retval == -1:
                print("Error Code  : ", retval)
                _raise_driver_error("write_numba_oneapi_mem_buffer_to_device",
                                    -1)
            return dArr
        else:
            _raise_unsupported_type_error("copy_array_to_device")

    def copy_array_from_device(self, array):
        if not isinstance(array, DeviceArray):
            _raise_unsupported_type_error("copy_array_to_device")
        retval = (_numba_oneapi_pybindings
                  .lib
                  .read_numba_oneapi_mem_buffer_from_device(
                      self._queue_ptr,
                      array.get_buffer_obj()[0],
                      True,
                      0,
                      array.get_buffer_size(),
                      array.get_data_ptr()))
        if retval == -1:
            print("Error Code  : ", retval)
            _raise_driver_error("read_numba_oneapi_mem_buffer_from_device", -1)


################################## Runtime class #########################
class _Runtime():
    """Runtime is a singleton class that creates a numba_oneapi_runtime
    object. The numba_oneapi_runtime (runtime) object on creation
    instantiates a OpenCL context and a corresponding OpenCL command
    queue for the first available CPU on the system. Similarly, the
    runtime object also stores the context and command queue for the
    first available GPU on the system.
    """
    _runtime = None
    _cpu_device = None
    _gpu_device = None

    def __new__(cls):
        obj = cls._runtime
        if obj is not None:
            return obj
        else:
            obj = object.__new__(cls)
            ffiobj = _numba_oneapi_pybindings.ffi.new("runtime_t *")
            retval = (_numba_oneapi_pybindings
                      .lib
                      .create_numba_oneapi_runtime(ffiobj))
            if(retval):
                print("Error Code  : ", retval)
                _raise_driver_error("create_numba_oneapi_runtime", -1)

            cls._runtime = ffiobj

            if cls._runtime[0][0].has_cpu:
                cls._cpu_device = Device(
                    cls._runtime[0][0].first_cpu_device.device,
                    cls._runtime[0][0].first_cpu_device.context,
                    cls._runtime[0][0].first_cpu_device.queue)
            else:
                # What should we do here? Raise an exception? Provide warning?
                # Maybe do not do anything here, only when this context is to
                # be used then first check if the context is populated.
                print("No CPU device")

            if cls._runtime[0][0].has_gpu:
                cls._gpu_device = Device(
                    cls._runtime[0][0].first_gpu_device.device,
                    cls._runtime[0][0].first_gpu_device.context,
                    cls._runtime[0][0].first_gpu_device.queue)
            else:
                # Same as the cpu case above.
                print("No GPU device")

        return obj

    def __init__(self):
        pass

    def __del__(self):
        print("Delete numba_oneapi_runtime object.")
        retval = (_numba_oneapi_pybindings
                  .lib
                  .destroy_numba_oneapi_runtime(_Runtime._runtime))
        if(retval):
            _raise_driver_error("destroy_numba_oneapi_runtime", -1)

    def get_cpu_device(self):
        if(self._cpu_device is None):
            _raise_device_not_found_error("get_cpu_device")

        return self._cpu_device

    def get_gpu_device(self):
        if(self._gpu_device is None):
            _raise_device_not_found_error("get_gpu_device")

        return self._gpu_device


runtime = _Runtime()
