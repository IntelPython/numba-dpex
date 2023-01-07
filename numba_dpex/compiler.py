# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

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
from numba.core.compiler_lock import global_compiler_lock

from numba_dpex import config
from numba_dpex.core.compiler import Compiler
from numba_dpex.core.exceptions import KernelHasReturnValueError
from numba_dpex.core.types import Array  # , USMNdArray

from . import spirv_generator

# from numba_dpex.core.utils import get_info_from_suai
# from numba_dpex.utils import (
#     as_usm_obj,
#     copy_from_numpy_to_usm_obj,
#     copy_to_numpy_from_usm_obj,
#     has_usm_memory,
#     mix_datatype_err_msg,
# )


_RO_KERNEL_ARG = "read_only"
_WO_KERNEL_ARG = "write_only"
_RW_KERNEL_ARG = "read_write"


def _raise_invalid_kernel_enqueue_args():
    error_message = (
        "Incorrect number of arguments for enqueuing numba_dpex.kernel. "
        "Usage: device_env, global size, local size. "
        "The local size argument is optional."
    )
    raise ValueError(error_message)


@global_compiler_lock
def compile_with_depx(pyfunc, return_type, args, is_kernel, debug=None):
    """
    Compiles the function using the dpex compiler pipeline and returns the
    compiled result.

    Args:
        pyfunc: The Python function to be compiled.
        return_type: The Numba type of the return value.
        args: The list of arguments sent to the Python function.
        is_kernel (bool): Indicates whether the function is decorated
        with @numba_depx.kernel or not.
        debug (bool): Flag to turn debug mode ON/OFF.

    Returns:
        cres: Compiled result.

    Raises:
        TypeError: @numba_depx.kernel does not allow users to return any
            value. TypeError is raised when users do.

    """
    # First compilation will trigger the initialization of the backend.
    from .core.descriptor import dpex_target

    typingctx = dpex_target.typing_context
    targetctx = dpex_target.target_context

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
            pipeline_class=Compiler,
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
            pipeline_class=Compiler,
        )
    else:
        assert 0

    if (
        is_kernel
        and cres.signature.return_type is not None
        and cres.signature.return_type != types.void
    ):
        raise KernelHasReturnValueError(
            kernel_name=pyfunc.__name__, return_type=cres.signature.return_type
        )

    # Linking depending libraries
    library = cres.library
    library.finalize()

    return cres


def compile_kernel_parfor(
    sycl_queue, func_ir, args, args_with_addrspaces, debug=None
):
    # We only accept numba_dpex.core.types.Array type
    for arg in args_with_addrspaces:
        if isinstance(arg, types.npytypes.Array) and not isinstance(arg, Array):
            raise TypeError(
                "Only numba_dpex.core.types.Array objects are supported as "
                + "kernel arguments. Received %s" % (type(arg))
            )
    if config.DEBUG:
        print("compile_kernel_parfor", args)
        for a in args_with_addrspaces:
            print(a, type(a))
            if isinstance(a, types.npytypes.Array):
                print("addrspace:", a.addrspace)

    cres = compile_with_depx(
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
    oclkern = Kernel(
        context=cres.target_context,
        sycl_queue=sycl_queue,
        llvm_module=kernel.module,
        name=kernel.name,
        argtypes=args_with_addrspaces,
    )

    return oclkern


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


class KernelBase(object):
    """Define interface for configurable kernels"""

    def __init__(self):
        self.global_size = []
        self.local_size = []
        self.sycl_queue = None

        # list of supported access types, stored in dict for fast lookup
        self.valid_access_types = {
            _RO_KERNEL_ARG: _RO_KERNEL_ARG,
            _WO_KERNEL_ARG: _WO_KERNEL_ARG,
            _RW_KERNEL_ARG: _RW_KERNEL_ARG,
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


class Kernel(KernelBase):
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
        super(Kernel, self).__init__()
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

    # def __call__(self, *args):
    #     """
    #     Create a list of the kernel arguments by unpacking pyobject values
    #     into ctypes values.
    #     """

    #     kernelargs = []
    #     internal_device_arrs = []
    #     for ty, val, access_type in zip(
    #         self.argument_types, args, self.ordered_arg_access_types
    #     ):
    #         self._unpack_argument(
    #             ty,
    #             val,
    #             self.sycl_queue,
    #             kernelargs,
    #             internal_device_arrs,
    #             access_type,
    #         )

    #     self.sycl_queue.submit(
    #         self.kernel, kernelargs, self.global_size, self.local_size
    #     )
    #     self.sycl_queue.wait()

    #     for ty, val, i_dev_arr, access_type in zip(
    #         self.argument_types,
    #         args,
    #         internal_device_arrs,
    #         self.ordered_arg_access_types,
    #     ):
    #         self._pack_argument(
    #             ty, val, self.sycl_queue, i_dev_arr, access_type
    #         )

    # def _pack_argument(self, ty, val, sycl_queue, device_arr, access_type):
    #     """
    #     Copy device data back to host
    #     """
    #     if device_arr and (
    #         access_type not in self.valid_access_types
    #         or access_type in self.valid_access_types
    #         and self.valid_access_types[access_type] != _RO_KERNEL_ARG
    #     ):
    #         # We copy the data back from usm allocated data
    #         # container to original data container.
    #         usm_mem, orig_ndarr, packed_ndarr, packed = device_arr
    #         copy_to_numpy_from_usm_obj(usm_mem, packed_ndarr)
    #         if packed:
    #             np.copyto(orig_ndarr, packed_ndarr)

    # def _unpack_device_array_argument(
    #     self, size, itemsize, buf, shape, strides, ndim, kernelargs
    # ):
    #     """
    #     Implements the unpacking logic for array arguments.

    #     Args:
    #         size: Total number of elements in the array.
    #         itemsize: Size in bytes of each element in the array.
    #         buf: The pointer to the memory.
    #         shape: The shape of the array.
    #         ndim: Number of dimension.
    #         kernelargs: Array where the arguments of the kernel is stored.
    #     """
    #     # meminfo
    #     kernelargs.append(ctypes.c_size_t(0))
    #     # parent
    #     kernelargs.append(ctypes.c_size_t(0))
    #     kernelargs.append(ctypes.c_longlong(size))
    #     kernelargs.append(ctypes.c_longlong(itemsize))
    #     kernelargs.append(buf)
    #     for ax in range(ndim):
    #         kernelargs.append(ctypes.c_longlong(shape[ax]))
    #     for ax in range(ndim):
    #         kernelargs.append(ctypes.c_longlong(strides[ax]))

    # def _unpack_USMNdArray(self, val, kernelargs):
    #     (
    #         usm_mem,
    #         total_size,
    #         shape,
    #         ndim,
    #         itemsize,
    #         strides,
    #         dtype,
    #     ) = get_info_from_suai(val)

    #     self._unpack_device_array_argument(
    #         total_size,
    #         itemsize,
    #         usm_mem,
    #         shape,
    #         strides,
    #         ndim,
    #         kernelargs,
    #     )

    # def _unpack_Array(
    #     self, val, sycl_queue, kernelargs, device_arrs, access_type
    # ):
    #     packed_val = val
    #     usm_mem = has_usm_memory(val)
    #     if usm_mem is None:
    #         default_behavior = self.check_for_invalid_access_type(access_type)
    #         usm_mem = as_usm_obj(val, queue=sycl_queue, copy=False)

    #         orig_val = val
    #         packed = False
    #         if not val.flags.c_contiguous:
    #             # If the numpy.ndarray is not C-contiguous
    #             # we pack the strided array into a packed array.
    #             # This allows us to treat the data from here on as C-contiguous.
    #             # While packing we treat the data as C-contiguous.
    #             # We store the reference of both (strided and packed)
    #             # array and during unpacking we use numpy.copyto() to copy
    #             # the data back from the packed temporary array to the
    #             # original strided array.
    #             packed_val = val.flatten(order="C")
    #             packed = True

    #         if (
    #             default_behavior
    #             or self.valid_access_types[access_type] == _RO_KERNEL_ARG
    #             or self.valid_access_types[access_type] == _RW_KERNEL_ARG
    #         ):
    #             copy_from_numpy_to_usm_obj(usm_mem, packed_val)

    #         device_arrs[-1] = (usm_mem, orig_val, packed_val, packed)

    #     self._unpack_device_array_argument(
    #         packed_val.size,
    #         packed_val.dtype.itemsize,
    #         usm_mem,
    #         packed_val.shape,
    #         packed_val.strides,
    #         packed_val.ndim,
    #         kernelargs,
    #     )

    # def _unpack_argument(
    #     self, ty, val, sycl_queue, kernelargs, device_arrs, access_type
    # ):
    #     """
    #     Unpacks the arguments that are to be passed to the SYCL kernel from
    #     Numba types to Ctypes.

    #     Args:
    #         ty: The data types of the kernel argument defined as in instance of
    #             numba.types.
    #         val: The value of the kernel argument.
    #         sycl_queue (dpctl.SyclQueue): A ``dpctl.SyclQueue`` object. The
    #             queue object will be used whenever USM memory allocation is
    #             needed during unpacking of an numpy.ndarray argument.
    #         kernelargs (list): The list of kernel arguments into which the
    #             current kernel argument will be appended.
    #         device_arrs (list): A list of tuples that is used to store the
    #             triples corresponding to the USM memorry allocated for an
    #             ``numpy.ndarray`` argument, a wrapper ``ndarray`` created from
    #             the USM memory, and the original ``ndarray`` argument.
    #         access_type : The type of access for an array argument.

    #     Raises:
    #         NotImplementedError: If the type of argument is not yet supported,
    #             then a ``NotImplementedError`` is raised.

    #     """

    #     device_arrs.append(None)

    #     if isinstance(ty, USMNdArray):
    #         self._unpack_USMNdArray(val, kernelargs)
    #     elif isinstance(ty, types.Array):
    #         self._unpack_Array(
    #             val, sycl_queue, kernelargs, device_arrs, access_type
    #         )
    #     elif ty == types.int64:
    #         cval = ctypes.c_longlong(val)
    #         kernelargs.append(cval)
    #     elif ty == types.uint64:
    #         cval = ctypes.c_ulonglong(val)
    #         kernelargs.append(cval)
    #     elif ty == types.int32:
    #         cval = ctypes.c_int(val)
    #         kernelargs.append(cval)
    #     elif ty == types.uint32:
    #         cval = ctypes.c_uint(val)
    #         kernelargs.append(cval)
    #     elif ty == types.float64:
    #         cval = ctypes.c_double(val)
    #         kernelargs.append(cval)
    #     elif ty == types.float32:
    #         cval = ctypes.c_float(val)
    #         kernelargs.append(cval)
    #     elif ty == types.boolean:
    #         cval = ctypes.c_uint8(int(val))
    #         kernelargs.append(cval)
    #     elif ty == types.complex64:
    #         raise NotImplementedError(ty, val)
    #     elif ty == types.complex128:
    #         raise NotImplementedError(ty, val)
    #     else:
    #         raise NotImplementedError(ty, val)

    # def check_for_invalid_access_type(self, access_type):
    #     if access_type not in self.valid_access_types:
    #         msg = (
    #             "[!] %s is not a valid access type. "
    #             "Supported access types are [" % (access_type)
    #         )
    #         for key in self.valid_access_types:
    #             msg += " %s |" % (key)

    #         msg = msg[:-1] + "]"
    #         if access_type is not None:
    #             print(msg)
    #         return True
    #     else:
    #         return False
