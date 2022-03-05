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

import copy
import ctypes
import warnings
from inspect import signature
from types import FunctionType

import dpctl
import dpctl.program as dpctl_prog
import dpctl.utils
import numpy as np
from numba.core import compiler, ir, types
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typing.templates import AbstractTemplate, ConcreteTemplate
from numba_dpex import config
from numba_dpex.dpctl_iface import USMNdArrayType
from numba_dpex.dpctl_support import dpctl_version
from numba_dpex.dppy_array_type import DPPYArray
from numba_dpex.dppy_parfor_diagnostics import ExtendedParforDiagnostics
from numba_dpex.utils import (
    IndeterminateExecutionQueueError,
    as_usm_obj,
    assert_no_return,
    cfd_ctx_mgr_wrng_msg,
    copy_from_numpy_to_usm_obj,
    copy_to_numpy_from_usm_obj,
    get_info_from_suai,
    has_usm_memory,
    mix_datatype_err_msg,
)

from . import spirv_generator
from .dppy_passbuilder import DPPYPassBuilder

_NUMBA_DPEX_READ_ONLY = "read_only"
_NUMBA_DPEX_WRITE_ONLY = "write_only"
_NUMBA_DPEX_READ_WRITE = "read_write"


def _raise_datatype_mixed_error(argtypes):
    error_message = mix_datatype_err_msg + ("%s" % str(argtypes))
    raise TypeError(error_message)


def _raise_no_device_found_error():
    error_message = (
        "No SYCL device specified. "
        "Usage : jit_fn[device, globalsize, localsize](...)"
    )
    raise ValueError(error_message)


def _raise_invalid_kernel_enqueue_args():
    error_message = (
        "Incorrect number of arguments for enquing dppy.kernel. "
        "Usage: device_env, global size, local size. "
        "The local size argument is optional."
    )
    raise ValueError(error_message)


def get_ordered_arg_access_types(pyfunc, access_types):
    # Construct a list of access type of each arg according to their position
    ordered_arg_access_types = []
    sig = signature(pyfunc, follow_wrapped=False)
    for idx, arg_name in enumerate(sig.parameters):
        if access_types:
            for key in access_types:
                if arg_name in access_types[key]:
                    ordered_arg_access_types.append(key)
        if len(ordered_arg_access_types) <= idx:
            ordered_arg_access_types.append(None)

    return ordered_arg_access_types


class DPPYCompiler(CompilerBase):
    """DPPY Compiler"""

    def define_pipelines(self):
        # this maintains the objmode fallback behaviour
        pms = []
        self.state.parfor_diagnostics = ExtendedParforDiagnostics()
        self.state.metadata[
            "parfor_diagnostics"
        ] = self.state.parfor_diagnostics
        if not self.state.flags.force_pyobject:
            # print("numba-dpex [INFO]: Using numba-dpex pipeline")
            pms.append(DPPYPassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(
                DefaultPassBuilder.define_objectmode_pipeline(self.state)
            )
        return pms


@global_compiler_lock
def compile_with_dppy(pyfunc, return_type, args, is_kernel, debug=None):
    """
    Compiles with Numba_dppy's pipeline and returns the compiled result.

    Args:
        pyfunc: The Python function to be compiled.
        return_type: The Numba type of the return value.
        args: The list of arguments sent to the Python function.
        is_kernel (bool): Indicates whether the function is decorated
            with @dppy.kernel or not.
        debug (bool): Flag to turn debug mode ON/OFF.

    Returns:
        cres: Compiled result.

    Raises:
        TypeError: @dppy.kernel does not allow users to return any
            value. TypeError is raised when users do.

    """
    # First compilation will trigger the initialization of the OpenCL backend.
    from .descriptor import dppy_target

    typingctx = dppy_target.typing_context
    targetctx = dppy_target.target_context

    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.debuginfo = config.DEBUGINFO_DEFAULT
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.nrt = False

    if debug is not None:
        flags.debuginfo = debug

    # Run compilation pipeline
    if isinstance(pyfunc, FunctionType):
        cres = compiler.compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=DPPYCompiler,
        )
    elif isinstance(pyfunc, ir.FunctionIR):
        cres = compiler.compile_ir(
            typingctx=typingctx,
            targetctx=targetctx,
            func_ir=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=DPPYCompiler,
        )
    else:
        assert 0

    if is_kernel:
        assert_no_return(cres.signature.return_type)

    # Linking depending libraries
    library = cres.library
    library.finalize()

    return cres


def compile_kernel(sycl_queue, pyfunc, args, access_types, debug=None):
    # For any array we only accept numba_dpex.dppy_array_type.DPPYArray
    for arg in args:
        if isinstance(arg, types.npytypes.Array) and not isinstance(
            arg, DPPYArray
        ):
            raise TypeError(
                "We only accept DPPYArray as type of array-like objects. We received %s"
                % (type(arg))
            )

    if config.DEBUG:
        print("compile_kernel", args)
        debug = True
    if not sycl_queue:
        # We expect the sycl_queue to be provided when this function is called
        raise ValueError("SYCL queue is required for compiling a kernel")

    cres = compile_with_dppy(
        pyfunc=pyfunc, return_type=None, args=args, is_kernel=True, debug=debug
    )
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)

    # A reference to the target context is stored in the DPPYKernel to
    # reference the context later in code generation. For example, we link
    # the kernel object with a spir_func defining atomic operations only
    # when atomic operations are used in the kernel.
    oclkern = DPPYKernel(
        context=cres.target_context,
        sycl_queue=sycl_queue,
        llvm_module=kernel.module,
        name=kernel.name,
        argtypes=cres.signature.args,
        ordered_arg_access_types=access_types,
    )
    return oclkern


def compile_kernel_parfor(
    sycl_queue, func_ir, args, args_with_addrspaces, debug=None
):
    # For any array we only accept numba_dpex.dppy_array_type.DPPYArray
    for arg in args_with_addrspaces:
        if isinstance(arg, types.npytypes.Array) and not isinstance(
            arg, DPPYArray
        ):
            raise TypeError(
                "We only accept DPPYArray as type of array-like objects. We received %s"
                % (type(arg))
            )
    if config.DEBUG:
        print("compile_kernel_parfor", args)
        for a in args_with_addrspaces:
            print(a, type(a))
            if isinstance(a, types.npytypes.Array):
                print("addrspace:", a.addrspace)

    cres = compile_with_dppy(
        pyfunc=func_ir,
        return_type=None,
        args=args_with_addrspaces,
        is_kernel=True,
        debug=debug,
    )
    func = cres.library.get_function(cres.fndesc.llvm_func_name)

    if config.DEBUG:
        print("compile_kernel_parfor signature", cres.signature.args)
        for a in cres.signature.args:
            print(a, type(a))

    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    oclkern = DPPYKernel(
        context=cres.target_context,
        sycl_queue=sycl_queue,
        llvm_module=kernel.module,
        name=kernel.name,
        argtypes=args_with_addrspaces,
    )

    return oclkern


def compile_dppy_func(pyfunc, return_type, args, debug=None):
    cres = compile_with_dppy(
        pyfunc=pyfunc,
        return_type=return_type,
        args=args,
        is_kernel=False,
        debug=debug,
    )
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    cres.target_context.mark_ocl_device(func)
    devfn = DPPYFunction(cres)

    class dppy_function_template(ConcreteTemplate):
        key = devfn
        cases = [cres.signature]

    cres.typing_context.insert_user_function(devfn, dppy_function_template)
    libs = [cres.library]
    cres.target_context.insert_user_function(devfn, cres.fndesc, libs)
    return devfn


# Compile dppy function template
def compile_dppy_func_template(pyfunc, debug=None):
    """Compile a DPPYFunctionTemplate"""
    from .descriptor import dppy_target

    dft = DPPYFunctionTemplate(pyfunc, debug=debug)

    class dppy_function_template(AbstractTemplate):
        key = dft

        def generic(self, args, kws):
            assert not kws
            return dft.compile(args)

    typingctx = dppy_target.typing_context
    typingctx.insert_user_function(dft, dppy_function_template)
    return dft


class DPPYFunctionTemplate(object):
    """Unmaterialized dppy function"""

    def __init__(self, pyfunc, debug=None):
        self.py_func = pyfunc
        self.debug = debug
        # self.inline = inline
        self._compileinfos = {}

    def compile(self, args):
        """Compile the function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.
        """
        if args not in self._compileinfos:
            cres = compile_with_dppy(
                pyfunc=self.py_func,
                return_type=None,
                args=args,
                is_kernel=False,
                debug=self.debug,
            )
            func = cres.library.get_function(cres.fndesc.llvm_func_name)
            cres.target_context.mark_ocl_device(func)
            first_definition = not self._compileinfos
            self._compileinfos[args] = cres
            libs = [cres.library]

            if first_definition:
                # First definition
                cres.target_context.insert_user_function(
                    self, cres.fndesc, libs
                )
            else:
                cres.target_context.add_user_function(self, cres.fndesc, libs)

        else:
            cres = self._compileinfos[args]

        return cres.signature


class DPPYFunction(object):
    def __init__(self, cres):
        self.cres = cres


def _ensure_valid_work_item_grid(val, sycl_queue):

    if not isinstance(val, (tuple, list, int)):
        error_message = (
            "Cannot create work item dimension from provided argument"
        )
        raise ValueError(error_message)

    if isinstance(val, int):
        val = [val]

    # TODO: we need some way to check the max dimensions
    """
    if len(val) > device_env.get_max_work_item_dims():
        error_message = ("Unsupported number of work item dimensions ")
        raise ValueError(error_message)
    """

    return list(
        val[::-1]
    )  # reversing due to sycl and opencl interop kernel range mismatch semantic


def _ensure_valid_work_group_size(val, work_item_grid):

    if not isinstance(val, (tuple, list, int)):
        error_message = (
            "Cannot create work item dimension from provided argument"
        )
        raise ValueError(error_message)

    if isinstance(val, int):
        val = [val]

    if len(val) != len(work_item_grid):
        error_message = (
            "Unsupported number of work item dimensions, "
            + "dimensions of global and local work items has to be the same "
        )
        raise ValueError(error_message)

    return list(
        val[::-1]
    )  # reversing due to sycl and opencl interop kernel range mismatch semantic


class DPPYKernelBase(object):
    """Define interface for configurable kernels"""

    def __init__(self):
        self.global_size = []
        self.local_size = []
        self.sycl_queue = None

        # list of supported access types, stored in dict for fast lookup
        self.valid_access_types = {
            _NUMBA_DPEX_READ_ONLY: _NUMBA_DPEX_READ_ONLY,
            _NUMBA_DPEX_WRITE_ONLY: _NUMBA_DPEX_WRITE_ONLY,
            _NUMBA_DPEX_READ_WRITE: _NUMBA_DPEX_READ_WRITE,
        }

    def copy(self):
        return copy.copy(self)

    def configure(self, sycl_queue, global_size, local_size=None):
        """Configure the OpenCL kernel. The local_size can be None"""
        clone = self.copy()
        clone.global_size = global_size
        clone.local_size = local_size
        clone.sycl_queue = sycl_queue

        return clone

    def __getitem__(self, args):
        """Mimick CUDA python's square-bracket notation for configuration.
        This assumes the argument to be:
            `global size, local size`
        """
        ls = None
        nargs = len(args)
        # Check if the kernel enquing arguments are sane
        if nargs < 1 or nargs > 2:
            _raise_invalid_kernel_enqueue_args

        sycl_queue = dpctl.get_current_queue()

        gs = _ensure_valid_work_item_grid(args[0], sycl_queue)
        # If the optional local size argument is provided
        if nargs == 2 and args[1] != []:
            ls = _ensure_valid_work_group_size(args[1], gs)

        return self.configure(sycl_queue, gs, ls)


class DPPYKernel(DPPYKernelBase):
    """
    A OCL kernel object
    """

    def __init__(
        self,
        context,
        sycl_queue,
        llvm_module,
        name,
        argtypes,
        ordered_arg_access_types=None,
    ):
        super(DPPYKernel, self).__init__()
        self._llvm_module = llvm_module
        self.assembly = self.binary = llvm_module.__str__()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self.ordered_arg_access_types = ordered_arg_access_types
        self._argloc = []
        self.sycl_queue = sycl_queue
        self.context = context

        dpctl_create_program_from_spirv_flags = []
        # First-time compilation using SPIRV-Tools
        if config.DEBUG:
            with open("llvm_kernel.ll", "w") as f:
                f.write(self.binary)

        if config.DEBUG or config.OPT == 0:
            # if debug is ON we need to pass additional
            # flags to igc.
            dpctl_create_program_from_spirv_flags = ["-g", "-cl-opt-disable"]

        self.spirv_bc = spirv_generator.llvm_to_spirv(
            self.context, self.assembly, self._llvm_module.as_bitcode()
        )

        # create a program
        self.program = dpctl_prog.create_program_from_spirv(
            self.sycl_queue,
            self.spirv_bc,
            " ".join(dpctl_create_program_from_spirv_flags),
        )
        #  create a kernel
        self.kernel = self.program.get_sycl_kernel(self.entry_name)

    def __call__(self, *args):
        """
        Create a list of the kernel arguments by unpacking pyobject values
        into ctypes values.
        """

        kernelargs = []
        internal_device_arrs = []
        for ty, val, access_type in zip(
            self.argument_types, args, self.ordered_arg_access_types
        ):
            self._unpack_argument(
                ty,
                val,
                self.sycl_queue,
                kernelargs,
                internal_device_arrs,
                access_type,
            )

        self.sycl_queue.submit(
            self.kernel, kernelargs, self.global_size, self.local_size
        )
        self.sycl_queue.wait()

        for ty, val, i_dev_arr, access_type in zip(
            self.argument_types,
            args,
            internal_device_arrs,
            self.ordered_arg_access_types,
        ):
            self._pack_argument(
                ty, val, self.sycl_queue, i_dev_arr, access_type
            )

    def _pack_argument(self, ty, val, sycl_queue, device_arr, access_type):
        """
        Copy device data back to host
        """
        if device_arr and (
            access_type not in self.valid_access_types
            or access_type in self.valid_access_types
            and self.valid_access_types[access_type] != _NUMBA_DPEX_READ_ONLY
        ):
            # We copy the data back from usm allocated data
            # container to original data container.
            usm_mem, orig_ndarr, packed_ndarr, packed = device_arr
            copy_to_numpy_from_usm_obj(usm_mem, packed_ndarr)
            if packed:
                np.copyto(orig_ndarr, packed_ndarr)

    def _unpack_device_array_argument(
        self, size, itemsize, buf, shape, strides, ndim, kernelargs
    ):
        """
        Implements the unpacking logic for array arguments.

        Args:
            size: Total number of elements in the array.
            itemsize: Size in bytes of each element in the array.
            buf: The pointer to the memory.
            shape: The shape of the array.
            ndim: Number of dimension.
            kernelargs: Array where the arguments of the kernel is stored.
        """
        # meminfo
        kernelargs.append(ctypes.c_size_t(0))
        # parent
        kernelargs.append(ctypes.c_size_t(0))
        kernelargs.append(ctypes.c_longlong(size))
        kernelargs.append(ctypes.c_longlong(itemsize))
        kernelargs.append(buf)
        for ax in range(ndim):
            kernelargs.append(ctypes.c_longlong(shape[ax]))
        for ax in range(ndim):
            kernelargs.append(ctypes.c_longlong(strides[ax]))

    def _unpack_USMNdArrayType(self, val, kernelargs):
        (
            usm_mem,
            total_size,
            shape,
            ndim,
            itemsize,
            strides,
            dtype,
        ) = get_info_from_suai(val)

        self._unpack_device_array_argument(
            total_size,
            itemsize,
            usm_mem,
            shape,
            strides,
            ndim,
            kernelargs,
        )

    def _unpack_Array(
        self, val, sycl_queue, kernelargs, device_arrs, access_type
    ):
        packed_val = val
        usm_mem = has_usm_memory(val)
        if usm_mem is None:
            default_behavior = self.check_for_invalid_access_type(access_type)
            usm_mem = as_usm_obj(val, queue=sycl_queue, copy=False)

            orig_val = val
            packed = False
            if not val.flags.c_contiguous:
                # If the numpy.ndarray is not C-contiguous
                # we pack the strided array into a packed array.
                # This allows us to treat the data from here on as C-contiguous.
                # While packing we treat the data as C-contiguous.
                # We store the reference of both (strided and packed)
                # array and during unpacking we use numpy.copyto() to copy
                # the data back from the packed temporary array to the
                # original strided array.
                packed_val = val.flatten(order="C")
                packed = True

            if (
                default_behavior
                or self.valid_access_types[access_type] == _NUMBA_DPEX_READ_ONLY
                or self.valid_access_types[access_type]
                == _NUMBA_DPEX_READ_WRITE
            ):
                copy_from_numpy_to_usm_obj(usm_mem, packed_val)

            device_arrs[-1] = (usm_mem, orig_val, packed_val, packed)

        self._unpack_device_array_argument(
            packed_val.size,
            packed_val.dtype.itemsize,
            usm_mem,
            packed_val.shape,
            packed_val.strides,
            packed_val.ndim,
            kernelargs,
        )

    def _unpack_argument(
        self, ty, val, sycl_queue, kernelargs, device_arrs, access_type
    ):
        """
        Unpacks the arguments that are to be passed to the SYCL kernel from
        Numba types to Ctypes.

        Args:
            ty: The data types of the kernel argument defined as in instance of
                numba.types.
            val: The value of the kernel argument.
            sycl_queue (dpctl.SyclQueue): A ``dpctl.SyclQueue`` object. The
                queue object will be used whenever USM memory allocation is
                needed during unpacking of an numpy.ndarray argument.
            kernelargs (list): The list of kernel arguments into which the
                current kernel argument will be appended.
            device_arrs (list): A list of tuples that is used to store the
                triples corresponding to the USM memorry allocated for an
                ``numpy.ndarray`` argument, a wrapper ``ndarray`` created from
                the USM memory, and the original ``ndarray`` argument.
            access_type : The type of access for an array argument.

        Raises:
            NotImplementedError: If the type of argument is not yet supported,
                then a ``NotImplementedError`` is raised.

        """

        device_arrs.append(None)

        if isinstance(ty, USMNdArrayType):
            self._unpack_USMNdArrayType(val, kernelargs)
        elif isinstance(ty, types.Array):
            self._unpack_Array(
                val, sycl_queue, kernelargs, device_arrs, access_type
            )
        elif ty == types.int64:
            cval = ctypes.c_longlong(val)
            kernelargs.append(cval)
        elif ty == types.uint64:
            cval = ctypes.c_ulonglong(val)
            kernelargs.append(cval)
        elif ty == types.int32:
            cval = ctypes.c_int(val)
            kernelargs.append(cval)
        elif ty == types.uint32:
            cval = ctypes.c_uint(val)
            kernelargs.append(cval)
        elif ty == types.float64:
            cval = ctypes.c_double(val)
            kernelargs.append(cval)
        elif ty == types.float32:
            cval = ctypes.c_float(val)
            kernelargs.append(cval)
        elif ty == types.boolean:
            cval = ctypes.c_uint8(int(val))
            kernelargs.append(cval)
        elif ty == types.complex64:
            raise NotImplementedError(ty, val)
        elif ty == types.complex128:
            raise NotImplementedError(ty, val)
        else:
            raise NotImplementedError(ty, val)

    def check_for_invalid_access_type(self, access_type):
        if access_type not in self.valid_access_types:
            msg = (
                "[!] %s is not a valid access type. "
                "Supported access types are [" % (access_type)
            )
            for key in self.valid_access_types:
                msg += " %s |" % (key)

            msg = msg[:-1] + "]"
            if access_type is not None:
                print(msg)
            return True
        else:
            return False


class JitDPPYKernel(DPPYKernelBase):
    def __init__(self, func, debug, access_types):

        super(JitDPPYKernel, self).__init__()

        self.py_func = func
        self.definitions = {}
        self.debug = debug
        self.access_types = access_types

        from .descriptor import dppy_target

        self.typingctx = dppy_target.typing_context

    def _get_argtypes(self, *args):
        """
        Convenience function to get the type of each argument.
        """
        return tuple([self.typingctx.resolve_argument_type(a) for a in args])

    def _datatype_is_same(self, argtypes):
        """
        This function will determine if there is any argument of type array and
        in case there are multiple array types if they are all of the same type.

        Args:
            argtypes: Numba type for each argument passed to a JitDPPYKernel.

        Returns:
            array_type: None if there are no argument of type array, or the
                        Numba type in case there is array type argument.
            bool: True if no array type arguments or if all array type arguments
                  are of same Numba type, False otherwise.

        """
        array_type = None
        for i, argtype in enumerate(argtypes):
            arg_is_array_type = isinstance(
                argtype, USMNdArrayType
            ) or isinstance(argtype, types.Array)
            if array_type is None and arg_is_array_type:
                array_type = argtype
            elif (
                array_type is not None
                and arg_is_array_type
                and type(argtype) is not type(array_type)
            ):
                return None, False
        return array_type, True

    def __call__(self, *args, **kwargs):
        assert not kwargs, "Keyword Arguments are not supported"

        argtypes = self._get_argtypes(*args)
        compute_queue = None

        # Get the array type and whether all array are of same type or not
        array_type, uniform = self._datatype_is_same(argtypes)
        if not uniform:
            _raise_datatype_mixed_error(argtypes)

        if type(array_type) == USMNdArrayType:
            if dpctl.is_in_device_context():
                warnings.warn(cfd_ctx_mgr_wrng_msg)

            queues = []
            for i, argtype in enumerate(argtypes):
                if type(argtype) == USMNdArrayType:
                    memory = dpctl.memory.as_usm_memory(args[i])
                    if dpctl_version < (0, 12):
                        queue = memory._queue
                    else:
                        queue = memory.sycl_queue
                    queues.append(queue)

            # dpctl.utils.get_exeuction_queue() checks if the queues passed are equivalent and returns a
            # SYCL queue if they are equivalent and None if they are not.
            compute_queue = dpctl.utils.get_execution_queue(queues)
            if compute_queue is None:
                raise IndeterminateExecutionQueueError(
                    "Data passed as argument are not equivalent. Please "
                    "create dpctl.tensor.usm_ndarray with equivalent SYCL queue."
                )

        if compute_queue is None:
            try:
                compute_queue = dpctl.get_current_queue()
            except:
                _raise_no_device_found_error()

        kernel = self.specialize(argtypes, compute_queue)
        cfg = kernel.configure(
            kernel.sycl_queue, self.global_size, self.local_size
        )
        cfg(*args)

    def specialize(self, argtypes, queue):
        # We specialize for argtypes and queue. These two are used as key for
        # caching as well.
        assert queue is not None

        sycl_ctx = None
        kernel = None
        # we were previously using the _env_ptr of the device_env, the sycl_queue
        # should be sufficient to cache the compiled kernel for now, but we should
        # use the device type to cache such kernels.
        key_definitions = argtypes
        result = self.definitions.get(key_definitions)
        if result:
            sycl_ctx, kernel = result

        if sycl_ctx and sycl_ctx == queue.sycl_context:
            return kernel
        else:
            kernel = compile_kernel(
                queue, self.py_func, argtypes, self.access_types, self.debug
            )
            self.definitions[key_definitions] = (queue.sycl_context, kernel)
        return kernel
