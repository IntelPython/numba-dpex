# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Module that contains numba style wrapper around sycl kernel submit."""

import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple, Union

import dpctl
from llvmlite import ir as llvmir
from llvmlite.ir.builder import IRBuilder
from numba.core import cgutils, types
from numba.core.cpu import CPUContext
from numba.core.datamodel import DataModelManager
from numba.core.types.containers import UniTuple

from numba_dpex.core import config
from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.runtime.context import DpexRTContext
from numba_dpex.core.types import USMNdArray
from numba_dpex.core.types.kernel_api.local_accessor import LocalAccessorType
from numba_dpex.core.types.kernel_api.ranges import NdRangeType, RangeType
from numba_dpex.core.utils import cgutils_extra
from numba_dpex.core.utils.kernel_flattened_args_builder import (
    KernelFlattenedArgsBuilder,
)
from numba_dpex.dpctl_iface import libsyclinterface_bindings as sycl

MAX_SIZE_OF_SYCL_RANGE = 3

_ARRAY_ALIGN = 16

OPEN_CL_OPT_DISABLE_FLAG = "-cl-opt-disable"
L0_OPT_DISABLE_FLAG = "-g"


# TODO: probably not best place for it. Should be in kernel_dispatcher once we
# get merge experimental. Right now it will cause cyclic import
class SPIRVKernelModule(NamedTuple):
    """Represents SPIRV binary code and function name in this binary"""

    kernel_name: str
    kernel_bitcode: bytes


@dataclass
class _KernelLaunchIRArguments:  # pylint: disable=too-many-instance-attributes
    """List of kernel launch arguments used in sycl.dpctl_queue_submit_range and
    sycl.dpctl_queue_submit_ndrange."""

    sycl_kernel_ref: llvmir.Instruction = None
    sycl_queue_ref: llvmir.Instruction = None
    arg_list: llvmir.Instruction = None
    arg_ty_list: llvmir.Instruction = None
    total_kernel_args: llvmir.Instruction = None
    global_range: llvmir.Instruction = None
    local_range: llvmir.Instruction = None
    range_size: llvmir.Instruction = None
    dep_events: llvmir.Instruction = None
    dep_events_len: llvmir.Instruction = None

    def to_list(self):
        """Returns list of arguments in the right order to pass to
        sycl.dpctl_queue_submit_range or sycl.dpctl_queue_submit_ndrange."""
        res = [
            self.sycl_kernel_ref,
            self.sycl_queue_ref,
            self.arg_list,
            self.arg_ty_list,
            self.total_kernel_args,
            self.global_range,
        ]

        if self.local_range is not None:
            res += [self.local_range]

        res += [
            self.range_size,
            self.dep_events,
            self.dep_events_len,
        ]

        return res


@dataclass
class _KernelLaunchIRCachedArguments:
    """Arguments that are being used in KernelLaunchIRBuilder that are either
    intermediate structure of the KernelLaunchIRBuilder like llvm IR array
    stored as a python array of llvm IR values or llvm IR values that may be
    used as an input for builder functions.

    Main goal is to prevent passing same argument during build process several
    times and to avoid passing output of the builder as an argument for another
    build method."""

    arg_list: list[llvmir.Instruction] = None
    arg_ty_list: list[types.Type] = None
    device_event_ref: llvmir.Instruction = None


class KernelLaunchIRBuilder:
    """
    Helper class to build the LLVM IR for the submission of a kernel.

    The class generates LLVM IR inside the current LLVM module that is needed
    for submitting kernels. The LLVM Values that
    """

    def __init__(
        self,
        context: CPUContext,
        builder: IRBuilder,
        kernel_dmm: DataModelManager,
    ):
        """Create a KernelLauncher for the specified kernel.

        Args:
            context: A Numba target context that will be used to generate the
                     code.
            builder: An llvmlite IRBuilder instance used to generate LLVM IR.
        """
        self.context = context
        self.builder = builder
        self.arguments = _KernelLaunchIRArguments()
        self.cached_arguments = _KernelLaunchIRCachedArguments()
        self.kernel_dmm = kernel_dmm
        self._cleanups = []

    @cached_property
    def dpexrt(self):
        """Dpex runtime context."""

        return DpexRTContext(self.context)

    def _build_nullptr(self):
        """Builds the LLVM IR to represent a null pointer.

        Returns: An LLVM Value storing a null pointer
        """
        zero = cgutils.alloca_once(
            self.builder, cgutils_extra.LLVMTypes.int64_t
        )
        self.builder.store(self.context.get_constant(types.int64, 0), zero)
        return self.builder.bitcast(
            zero,
            cgutils_extra.get_llvm_type(
                context=self.context, type=types.voidptr
            ),
        )

    # TODO: remove, not part of the builder
    def get_queue(self, exec_queue: dpctl.SyclQueue) -> llvmir.Instruction:
        """Allocates memory on the stack to store a DPCTLSyclQueueRef.

        Returns: A LLVM Value storing the pointer to the SYCL queue created
        using the filter string for the Python exec_queue (dpctl.SyclQueue).
        """

        # Allocate a stack var to store the queue created from the filter string
        sycl_queue_val = cgutils.alloca_once(
            self.builder,
            cgutils_extra.get_llvm_type(
                context=self.context, type=types.voidptr
            ),
        )
        # Insert a global constant to store the filter string
        device = self.context.insert_const_string(
            self.builder.module, exec_queue.sycl_device.filter_string
        )
        # Store the queue returned by DPEXRTQueue_CreateFromFilterString in a
        # local variable
        self.builder.store(
            self.dpexrt.get_queue_from_filter_string(
                builder=self.builder, device=device
            ),
            sycl_queue_val,
        )
        return self.builder.load(sycl_queue_val)

    def _allocate_array(
        self, numba_type: types.Type, size: int
    ) -> llvmir.Instruction:
        """Allocates an LLVM array of given type and size.

        Args:
            numba_type: type of the array to allocate,
            size: The size of the array to allocate.

        Returns: An LLVM IR value pointing to the array.
        """
        array = cgutils.alloca_once(
            self.builder,
            self.context.get_value_type(numba_type),
            size=self.context.get_constant(types.uintp, size),
        )
        array.align = _ARRAY_ALIGN
        return array

    def _populate_array_from_python_list(
        self,
        numba_type: types.Type,
        py_array: list[llvmir.Instruction],
        ll_array: llvmir.Instruction,
        force_cast: bool = False,
    ):
        """Populates LLVM values from an input Python list into an LLVM array.

        Args:
            numba_type: type of the array to allocate,
            py_array: array of llvm ir values to populate.
            ll_array: llvm ir value that represents an array to populate,
            force_cast: either force cast values to the provided type.
        """
        for idx, ll_value in enumerate(py_array):
            ll_array_dst = self.builder.gep(
                ll_array,
                [self.context.get_constant(types.int32, idx)],
            )
            # bitcast may be extra, but won't hurt,
            if force_cast:
                ll_value = self.builder.bitcast(
                    ll_value,
                    self.context.get_value_type(numba_type),
                )
            self.builder.store(ll_value, ll_array_dst)

    def _create_ll_from_py_list(
        self,
        numba_type: types.Type,
        list_of_ll_values: list[llvmir.Instruction],
        force_cast: bool = False,
    ) -> llvmir.Instruction:
        """Allocates an LLVM IR array of the same size as the input python list
        of LLVM IR Values and populates the array with the LLVM Values in the
        list.

        Args:
            numba_type: type of the array to allocate,
            list_of_ll_values: list of LLVM IR values to populate,
            force_cast: either force cast values to the provided type.

        Returns: An LLVM IR value pointing to the array.
        """
        ll_array = self._allocate_array(numba_type, len(list_of_ll_values))
        self._populate_array_from_python_list(
            numba_type, list_of_ll_values, ll_array, force_cast
        )

        return ll_array

    def _create_sycl_range(self, idx_range):
        """Allocate an array to store the extents of a sycl::range.

        Sycl supports upto 3-dimensional ranges and a such the array is
        statically sized to length three. Only the elements that store an actual
        range value are populated based on the size of the idx_range argument.

        """
        int64_range = [
            (
                self.builder.sext(rext, cgutils_extra.LLVMTypes.int64_t)
                if rext.type != cgutils_extra.LLVMTypes.int64_t
                else rext
            )
            for rext in idx_range
        ]

        # Index inversion is done here as numba-dpex first compiles a native
        # kernel (OpenCL or Level Zero) and then generates a SYCL
        # interoperability kernel from it. The convention for unit stride
        # dimensions is opposite for OpenCL and SYCL
        # refer:
        # https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:opencl:kernel-conventions-sycl
        # For this reason, although numba-dpex follows SYCL like indexing in
        # the kernel front-end while launching the kernel the indexing is
        # reversed.
        #
        # TODO[1]: It needs to be investigated if we need the index reversal
        # if we use SYCL-like LLVM IR indexing intrinsic instead of
        # OpenCL-like LLVM IR intrinsic functions.
        #
        # TODO[2]: Do we need to do this when the backend is LevelZero
        int64_range.reverse()

        return self._create_ll_from_py_list(types.uintp, int64_range)

    def set_kernel(self, sycl_kernel_ref: llvmir.Instruction):
        """Sets kernel to the argument list."""
        self.arguments.sycl_kernel_ref = sycl_kernel_ref

    def set_kernel_from_spirv(
        self, kernel_module: SPIRVKernelModule, debug=False
    ):
        """Sets kernel to the argument list from the SPIRV bytecode.

        It pastes bytecode as a constant string and create kernel bundle from it
        using SYCL API. It caches kernel, so it won't be sent to device second
        time.
        """
        # Inserts a global constant byte string in the current LLVM module to
        # store the passed in SPIR-V binary blob.
        queue_ref = self.arguments.sycl_queue_ref

        kernel_bc_byte_str = self.context.insert_const_bytes(
            self.builder.module,
            bytes=kernel_module.kernel_bitcode,
        )

        kernel_name = self.context.insert_const_string(
            self.builder.module, kernel_module.kernel_name
        )

        context_ref = sycl.dpctl_queue_get_context(self.builder, queue_ref)
        device_ref = sycl.dpctl_queue_get_device(self.builder, queue_ref)

        build_kernel_options = ""
        if debug:
            build_kernel_options = (
                OPEN_CL_OPT_DISABLE_FLAG + " " + L0_OPT_DISABLE_FLAG
            )
        if config.BUILD_KERNEL_OPTIONS:
            # User settings are higher priority than kernel configuration
            build_kernel_options = config.BUILD_KERNEL_OPTIONS
            if debug and not (
                OPEN_CL_OPT_DISABLE_FLAG in build_kernel_options
                and L0_OPT_DISABLE_FLAG in build_kernel_options
            ):
                warnings.warn(
                    "Debugging without device optimization may lead to "
                    "unexpected behavior"
                )

        if build_kernel_options != "":
            spv_compiler_options = self.context.insert_const_string(
                self.builder.module, build_kernel_options
            )
        else:
            spv_compiler_options = self.builder.load(
                cgutils_extra.create_null_ptr(self.builder, self.context)
            )

        # build_or_get_kernel steals reference to context and device cause it
        # needs to keep them alive for keys.
        kernel_ref = self.dpexrt.build_or_get_kernel(
            self.builder,
            [
                context_ref,
                device_ref,
                llvmir.Constant(
                    llvmir.IntType(64), hash(kernel_module.kernel_bitcode)
                ),
                kernel_bc_byte_str,
                llvmir.Constant(
                    llvmir.IntType(64), len(kernel_module.kernel_bitcode)
                ),
                spv_compiler_options,
                kernel_name,
            ],
        )

        self._cleanups.append(self._clean_kernel_ref)
        self.set_kernel(kernel_ref)

    def _clean_kernel_ref(self):
        sycl.dpctl_kernel_delete(self.builder, self.arguments.sycl_kernel_ref)
        self.arguments.sycl_kernel_ref = None

    def set_queue(self, sycl_queue_ref: llvmir.Instruction):
        """Sets queue to the argument list."""
        self.arguments.sycl_queue_ref = sycl_queue_ref

    def set_queue_from_arguments(
        self,
    ):
        """Sets the sycl queue from the first USMNdArray argument provided
        earlier."""
        queue_ref = get_queue_from_llvm_values(
            self.context,
            self.builder,
            self.cached_arguments.arg_ty_list,
            self.cached_arguments.arg_list,
        )

        if queue_ref is None:
            raise ValueError("There are no arguments that contain queue")

        self.set_queue(queue_ref)

    def set_range(
        self,
        global_range: list,
        local_range: list = None,
    ):
        """Sets global and local range if provided to the argument list."""
        self.arguments.global_range = self._create_sycl_range(global_range)
        if local_range is not None and len(local_range) > 0:
            self.arguments.local_range = self._create_sycl_range(local_range)
        self.arguments.range_size = self.context.get_constant(
            types.uintp, len(global_range)
        )

    def set_range_from_indexer(
        self,
        ty_indexer_arg: Union[RangeType, NdRangeType],
        ll_index_arg: llvmir.BaseStructType,
    ):
        """Returns two lists of LLVM IR Values that hold the unboxed extents of
        a Python Range or NdRange object.
        """
        ndim = ty_indexer_arg.ndim
        global_range_extents = []
        local_range_extents = []
        indexer_datamodel = self.context.data_model_manager.lookup(
            ty_indexer_arg
        )

        if isinstance(ty_indexer_arg, RangeType):
            for dim_num in range(ndim):
                dim_pos = indexer_datamodel.get_field_position(
                    "dim" + str(dim_num)
                )
                global_range_extents.append(
                    self.builder.extract_value(ll_index_arg, dim_pos)
                )
        elif isinstance(ty_indexer_arg, NdRangeType):
            for dim_num in range(ndim):
                gdim_pos = indexer_datamodel.get_field_position(
                    "gdim" + str(dim_num)
                )
                global_range_extents.append(
                    self.builder.extract_value(ll_index_arg, gdim_pos)
                )
                ldim_pos = indexer_datamodel.get_field_position(
                    "ldim" + str(dim_num)
                )
                local_range_extents.append(
                    self.builder.extract_value(ll_index_arg, ldim_pos)
                )
        else:
            raise UnreachableError

        self.set_range(global_range_extents, local_range_extents)

    def set_arguments(
        self,
        ty_kernel_args: list[types.Type],
        kernel_args: list[llvmir.Instruction],
    ):
        """Sets flattened kernel args, kernel arg types and number of those
        arguments to the argument list."""
        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(
                self.builder,
                "DPEX-DEBUG: Populating kernel args and arg type arrays.\n",
            )

        self.cached_arguments.arg_ty_list = ty_kernel_args
        self.cached_arguments.arg_list = kernel_args

        num_flattened_kernel_args = self._get_num_flattened_kernel_args(
            kernel_argtys=ty_kernel_args,
        )

        # Create LLVM values for the kernel args list and kernel arg types list
        args_list = self._allocate_array(
            types.voidptr,
            num_flattened_kernel_args,
        )

        args_ty_list = self._allocate_array(
            types.int32,
            num_flattened_kernel_args,
        )

        kernel_args_ptrs = []
        for arg in kernel_args:
            with self.builder.goto_entry_block():
                ptr = self.builder.alloca(arg.type)
            self.builder.store(arg, ptr)
            kernel_args_ptrs.append(ptr)

        # Populate the args_list and the args_ty_list LLVM arrays
        self._populate_kernel_args_and_args_ty_arrays(
            host_callargs_ptrs=kernel_args_ptrs,
            host_kernel_argtys=ty_kernel_args,
            kernel_args_list=args_list,
            kernel_args_ty_list=args_ty_list,
        )

        self.arguments.arg_list = args_list
        self.arguments.arg_ty_list = args_ty_list
        self.arguments.total_kernel_args = self.context.get_constant(
            types.uintp, num_flattened_kernel_args
        )

    def _extract_llvm_values_from_tuple(
        self,
        ll_tuple: llvmir.Instruction,
    ) -> list[llvmir.Instruction]:
        """Extracts LLVM IR values from llvm tuple into python array."""

        llvm_values = []
        for pos in range(len(ll_tuple.type)):
            llvm_values.append(self.builder.extract_value(ll_tuple, pos))

        return llvm_values

    def set_arguments_form_tuple(
        self,
        ty_kernel_args_tuple: UniTuple,
        ll_kernel_args_tuple: llvmir.Instruction,
    ):
        """Sets flattened kernel args, kernel arg types and number of those
        arguments to the argument list based on the arguments stored in tuple.
        """
        kernel_args = self._extract_llvm_values_from_tuple(ll_kernel_args_tuple)
        self.set_arguments(ty_kernel_args_tuple, kernel_args)

    def set_dependent_events(self, dep_events: list[llvmir.Instruction]):
        """Sets dependent events to the argument list."""
        ll_dep_events = self._create_ll_from_py_list(types.voidptr, dep_events)
        self.arguments.dep_events = ll_dep_events
        self.arguments.dep_events_len = self.context.get_constant(
            types.uintp, len(dep_events)
        )

    def set_dependent_events_from_tuple(
        self,
        ty_dependent_events: UniTuple,
        ll_dependent_events: llvmir.Instruction,
    ):
        """Set's dependent events from tuple represented by LLVM IR.

        Args:
            ll_dependent_events: tuple of numba's data models.
        """
        if len(ty_dependent_events) == 0:
            self.set_dependent_events([])
            return

        ty_event = ty_dependent_events[0]
        dm_dependent_events = self._extract_llvm_values_from_tuple(
            ll_dependent_events
        )
        dependent_events = []
        for dm_dependent_event in dm_dependent_events:
            event_struct_proxy = cgutils.create_struct_proxy(ty_event)(
                self.context,
                self.builder,
                value=dm_dependent_event,
            )
            dependent_events.append(event_struct_proxy.event_ref)

        self.set_dependent_events(dependent_events)

    def submit(self) -> llvmir.Instruction:
        """Submits kernel by calling sycl.dpctl_queue_submit_range or
        sycl.dpctl_queue_submit_ndrange. Must be called after all arguments
        set."""
        args = self.arguments.to_list()

        if self.arguments.local_range is None:
            event_ref = sycl.dpctl_queue_submit_range(self.builder, *args)
        else:
            event_ref = sycl.dpctl_queue_submit_ndrange(self.builder, *args)

        self.cached_arguments.device_event_ref = event_ref

        for cleanup in self._cleanups:
            cleanup()

        return event_ref

    def _allocate_meminfo_array(
        self,
    ) -> tuple[int, list[llvmir.Instruction]]:
        """Allocates an LLVM array value to store each memory info from all
        kernel arguments. The array is the populated with the LLVM value for
        every meminfo of the kernel arguments.
        """
        kernel_args = self.cached_arguments.arg_list
        kernel_argtys = self.cached_arguments.arg_ty_list

        meminfos = []
        for arg_num, argtype in enumerate(kernel_argtys):
            llvm_val = kernel_args[arg_num]

            meminfos += [
                meminfo
                for ty, meminfo in self.context.nrt.get_meminfos(
                    self.builder, argtype, llvm_val
                )
            ]

        meminfo_list = self._create_ll_from_py_list(types.voidptr, meminfos)

        return len(meminfos), meminfo_list

    def acquire_meminfo_and_submit_release(
        self,
    ) -> llvmir.Instruction:
        """Schedule sycl host task to release nrt meminfo of the arguments used
        to run job. Use it to keep arguments alive during kernel execution."""
        queue_ref = self.arguments.sycl_queue_ref
        event_ref = self.cached_arguments.device_event_ref

        total_meminfos, meminfo_list = self._allocate_meminfo_array()

        event_ref_ptr = self.builder.alloca(event_ref.type)
        self.builder.store(event_ref, event_ref_ptr)

        status_ptr = cgutils.alloca_once(
            self.builder, self.context.get_value_type(types.uint64)
        )
        host_eref = self.dpexrt.acquire_meminfo_and_schedule_release(
            self.builder,
            [
                self.context.nrt.get_nrt_api(self.builder),
                queue_ref,
                meminfo_list,
                self.context.get_constant(types.uintp, total_meminfos),
                event_ref_ptr,
                self.context.get_constant(types.uintp, 1),
                status_ptr,
            ],
        )
        return host_eref

    def _get_num_flattened_kernel_args(
        self,
        kernel_argtys: tuple[types.Type, ...],
    ) -> int:
        """Returns number of flattened arguments of kernel data model based on
        the numba types. Flattens usm arrays and complex values."""
        num_flattened_kernel_args = 0
        for arg_type in kernel_argtys:
            if isinstance(arg_type, USMNdArray):
                datamodel = self.kernel_dmm.lookup(arg_type)
                num_flattened_kernel_args += datamodel.flattened_field_count
            elif arg_type in [types.complex64, types.complex128]:
                num_flattened_kernel_args += 2
            else:
                num_flattened_kernel_args += 1

        return num_flattened_kernel_args

    def _update_kernel_args_list(
        self, arg_num, kernel_arg, kernel_args_list, kernel_args_ty_list
    ):
        kernel_arg_dst = self.builder.gep(
            kernel_args_list,
            [self.context.get_constant(types.int32, arg_num)],
        )
        kernel_arg_ty_dst = self.builder.gep(
            kernel_args_ty_list,
            [self.context.get_constant(types.int32, arg_num)],
        )
        self.builder.store(kernel_arg.llvm_val, kernel_arg_dst)
        self.builder.store(kernel_arg.typeid, kernel_arg_ty_dst)

    def _populate_kernel_args_and_args_ty_arrays(
        self,
        host_kernel_argtys,
        host_callargs_ptrs,
        kernel_args_list,
        kernel_args_ty_list,
    ):
        """Populates the array of kernel arguments and the array of typeids that
        are passed to DpctlQueue_Submit when executing a kernel function.
        """
        args_builder = KernelFlattenedArgsBuilder(
            context=self.context,
            builder=self.builder,
            kernel_dmm=self.kernel_dmm,
        )

        for arg_num, arg_type in enumerate(host_kernel_argtys):
            args_builder.add_argument(
                arg_type=arg_type,
                arg_packed_llvm_val=host_callargs_ptrs[arg_num],
            )

        for kernel_arg_num, karg in enumerate(
            args_builder.get_kernel_arg_list()
        ):
            self._update_kernel_args_list(
                kernel_arg_num, karg, kernel_args_list, kernel_args_ty_list
            )


def get_queue_from_llvm_values(
    ctx: CPUContext,
    builder: IRBuilder,
    ty_kernel_args: list[types.Type],
    ll_kernel_args: list[llvmir.Instruction],
):
    """
    Get the sycl queue from the first USMNdArray argument. Prior passes
    before lowering make sure that compute-follows-data is enforceable
    for a specific call to a kernel. As such, at the stage of lowering
    the queue from the first USMNdArray argument can be extracted.
    """
    for arg_num, argty in enumerate(ty_kernel_args):
        if isinstance(argty, USMNdArray) and not isinstance(
            argty, LocalAccessorType
        ):
            llvm_val = ll_kernel_args[arg_num]
            datamodel = ctx.data_model_manager.lookup(argty)
            sycl_queue_attr_pos = datamodel.get_field_position("sycl_queue")
            queue_ref = builder.extract_value(llvm_val, sycl_queue_attr_pos)
            break

    return queue_ref
