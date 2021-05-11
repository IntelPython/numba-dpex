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

from __future__ import print_function, absolute_import
import copy
from collections import namedtuple

from .dppy_passbuilder import DPPYPassBuilder
from numba.core.typing.templates import ConcreteTemplate
from numba.core import types, compiler, ir
from numba.core.typing.templates import AbstractTemplate
import ctypes
from types import FunctionType
from inspect import signature

import dpctl
import dpctl.memory as dpctl_mem
import dpctl.program as dpctl_prog
import numpy as np

from . import spirv_generator

import os
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba_dppy.dppy_parfor_diagnostics import ExtendedParforDiagnostics


DEBUG = os.environ.get("NUMBA_DPPY_DEBUG", None)
_NUMBA_DPPY_READ_ONLY = "read_only"
_NUMBA_DPPY_WRITE_ONLY = "write_only"
_NUMBA_DPPY_READ_WRITE = "read_write"


def _raise_no_device_found_error():
    error_message = (
        "No OpenCL device specified. "
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
        self.state.metadata["parfor_diagnostics"] = self.state.parfor_diagnostics
        if not self.state.flags.force_pyobject:
            # print("Numba-DPPY [INFO]: Using Numba-DPPY pipeline")
            pms.append(DPPYPassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(DefaultPassBuilder.define_objectmode_pipeline(self.state))
        return pms


def compile_with_dppy(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the OpenCL backend.
    from .descriptor import dppy_target

    typingctx = dppy_target.typing_context
    targetctx = dppy_target.target_context

    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    if debug:
        flags.set("debuginfo")
    flags.set("no_compile")
    flags.set("no_cpython_wrapper")
    flags.unset("nrt")

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
    # Linking depending libraries
    # targetctx.link_dependencies(cres.llvm_module, cres.target_context.linking)
    library = cres.library
    library.finalize()

    return cres


def compile_kernel(sycl_queue, pyfunc, args, access_types, debug=False):
    if DEBUG:
        print("compile_kernel", args)
        debug = True
    if not sycl_queue:
        # This will be get_current_queue
        sycl_queue = dpctl.get_current_queue()

    cres = compile_with_dppy(pyfunc, None, args, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    # The kernel objet should have a reference to the target context it is compiled for.
    # This is needed as we intend to shape the behavior of the kernel down the line
    # depending on the target context. For example, we want to link our kernel object
    # with implementation containing atomic operations only when atomic operations
    # are being used in the kernel.
    oclkern = DPPYKernel(
        context=cres.target_context,
        sycl_queue=sycl_queue,
        llvm_module=kernel.module,
        name=kernel.name,
        argtypes=cres.signature.args,
        ordered_arg_access_types=access_types,
    )
    return oclkern


def compile_kernel_parfor(sycl_queue, func_ir, args, args_with_addrspaces, debug=False):
    if DEBUG:
        print("compile_kernel_parfor", args)
        for a in args_with_addrspaces:
            print(a, type(a))
            if isinstance(a, types.npytypes.Array):
                print("addrspace:", a.addrspace)

    cres = compile_with_dppy(func_ir, None, args_with_addrspaces, debug=debug)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)

    if DEBUG:
        print("compile_kernel_parfor signature", cres.signature.args)
        for a in cres.signature.args:
            print(a, type(a))
    #            if isinstance(a, types.npytypes.Array):
    #                print("addrspace:", a.addrspace)

    kernel = cres.target_context.prepare_ocl_kernel(func, cres.signature.args)
    # kernel = cres.target_context.prepare_ocl_kernel(func, args_with_addrspaces)
    oclkern = DPPYKernel(
        context=cres.target_context,
        sycl_queue=sycl_queue,
        llvm_module=kernel.module,
        name=kernel.name,
        argtypes=args_with_addrspaces,
    )
    # argtypes=cres.signature.args)
    return oclkern


def compile_dppy_func(pyfunc, return_type, args, debug=False):
    cres = compile_with_dppy(pyfunc, return_type, args, debug=debug)
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
def compile_dppy_func_template(pyfunc):
    """Compile a DPPYFunctionTemplate"""
    from .descriptor import dppy_target

    dft = DPPYFunctionTemplate(pyfunc)

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

    def __init__(self, pyfunc, debug=False):
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
            cres = compile_with_dppy(self.py_func, None, args, debug=self.debug)
            func = cres.library.get_function(cres.fndesc.llvm_func_name)
            cres.target_context.mark_ocl_device(func)
            first_definition = not self._compileinfos
            self._compileinfos[args] = cres
            libs = [cres.library]

            if first_definition:
                # First definition
                cres.target_context.insert_user_function(self, cres.fndesc, libs)
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
        error_message = "Cannot create work item dimension from " "provided argument"
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
        error_message = "Cannot create work item dimension from " "provided argument"
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
            _NUMBA_DPPY_READ_ONLY: _NUMBA_DPPY_READ_ONLY,
            _NUMBA_DPPY_WRITE_ONLY: _NUMBA_DPPY_WRITE_ONLY,
            _NUMBA_DPPY_READ_WRITE: _NUMBA_DPPY_READ_WRITE,
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
        # First-time compilation using SPIRV-Tools
        if DEBUG:
            with open("llvm_kernel.ll", "w") as f:
                f.write(self.binary)

        self.spirv_bc = spirv_generator.llvm_to_spirv(self.context, self.binary)

        # create a program
        self.program = dpctl_prog.create_program_from_spirv(
            self.sycl_queue, self.spirv_bc
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
            self._pack_argument(ty, val, self.sycl_queue, i_dev_arr, access_type)

    def _pack_argument(self, ty, val, sycl_queue, device_arr, access_type):
        """
        Copy device data back to host
        """
        if device_arr and (
            access_type not in self.valid_access_types
            or access_type in self.valid_access_types
            and self.valid_access_types[access_type] != _NUMBA_DPPY_READ_ONLY
        ):
            # we get the date back to host if have created a
            # device_array or if access_type of this device_array
            # is not of type read_only and read_write
            usm_buf, usm_ndarr, orig_ndarray = device_arr
            np.copyto(orig_ndarray, usm_ndarr)

    def _unpack_device_array_argument(self, val, kernelargs):
        """
        Implements the unpacking logic for array arguments.
        """
        # meminfo
        kernelargs.append(ctypes.c_size_t(0))
        # parent
        kernelargs.append(ctypes.c_size_t(0))
        kernelargs.append(ctypes.c_longlong(val.size))
        kernelargs.append(ctypes.c_longlong(val.dtype.itemsize))
        kernelargs.append(val.base)
        for ax in range(val.ndim):
            kernelargs.append(ctypes.c_longlong(val.shape[ax]))
        for ax in range(val.ndim):
            kernelargs.append(ctypes.c_longlong(val.strides[ax]))

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

        if isinstance(ty, types.Array):
            if hasattr(val.base, "__sycl_usm_array_interface__"):
                self._unpack_device_array_argument(val, kernelargs)
            else:
                default_behavior = self.check_for_invalid_access_type(access_type)

                usm_buf = dpctl_mem.MemoryUSMShared(
                    val.size * val.dtype.itemsize, queue=sycl_queue
                )
                usm_ndarr = np.ndarray(val.shape, buffer=usm_buf, dtype=val.dtype)

                if (
                    default_behavior
                    or self.valid_access_types[access_type] == _NUMBA_DPPY_READ_ONLY
                    or self.valid_access_types[access_type] == _NUMBA_DPPY_READ_WRITE
                ):
                    np.copyto(usm_ndarr, val)

                device_arrs[-1] = (usm_buf, usm_ndarr, val)
                self._unpack_device_array_argument(usm_ndarr, kernelargs)

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
            if access_type != None:
                print(msg)
            return True
        else:
            return False


class JitDPPYKernel(DPPYKernelBase):
    def __init__(self, func, access_types):

        super(JitDPPYKernel, self).__init__()

        self.py_func = func
        self.definitions = {}
        self.access_types = access_types

        from .descriptor import dppy_target

        self.typingctx = dppy_target.typing_context

    def __call__(self, *args, **kwargs):
        assert not kwargs, "Keyword Arguments are not supported"
        if self.sycl_queue is None:
            try:
                self.sycl_queue = dpctl.get_current_queue()
            except:
                _raise_no_device_found_error()

        kernel = self.specialize(*args)
        cfg = kernel.configure(self.sycl_queue, self.global_size, self.local_size)
        cfg(*args)

    def specialize(self, *args):
        argtypes = tuple([self.typingctx.resolve_argument_type(a) for a in args])
        q = None
        kernel = None
        # we were previously using the _env_ptr of the device_env, the sycl_queue
        # should be sufficient to cache the compiled kernel for now, but we should
        # use the device type to cache such kernels
        # key_definitions = (self.sycl_queue, argtypes)
        key_definitions = argtypes
        result = self.definitions.get(key_definitions)
        if result:
            q, kernel = result

        if q == self.sycl_queue:
            return kernel
        else:
            kernel = compile_kernel(
                self.sycl_queue, self.py_func, argtypes, self.access_types
            )
            self.definitions[key_definitions] = (self.sycl_queue, kernel)
        return kernel
