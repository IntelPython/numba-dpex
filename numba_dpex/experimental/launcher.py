# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides a helper function to call a numba_dpex.kernel decorated function
from either CPython or a numba_dpex.dpjit decorated function.
"""

from typing import NamedTuple, Union

import dpctl
from llvmlite import ir as llvmir
from numba.core import cgutils, cpu, types
from numba.core.datamodel import default_manager as numba_default_dmm
from numba.core.types.containers import UniTuple
from numba.core.types.functions import Dispatcher
from numba.extending import intrinsic

from numba_dpex import config, dpjit, utils
from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.runtime.context import DpexRTContext
from numba_dpex.core.types import (
    DpctlSyclEvent,
    DpnpNdArray,
    NdRangeType,
    RangeType,
)
from numba_dpex.core.utils import kernel_launcher as kl
from numba_dpex.dpctl_iface import libsyclinterface_bindings as sycl
from numba_dpex.dpctl_iface.wrappers import wrap_event_reference
from numba_dpex.experimental.kernel_dispatcher import (
    KernelDispatcher,
    _KernelModule,
)
from numba_dpex.utils import create_null_ptr


class LLRange(NamedTuple):
    """Analog of Range and NdRange but for the llvm ir values."""

    global_range_extents: list
    local_range_extents: list


class CallKernelBuilder:
    """
    Helper class to build LLVM IR for a numba function that calls kernel.
    """

    def __init__(
        self,
        codegen_targetctx: cpu.CPUContext,
        builder: llvmir.IRBuilder,
    ):
        self.context = codegen_targetctx
        self.builder = builder

        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(
                self.builder,
                "DPEX-DEBUG: Inside the kernel launcher function\n",
            )

    def extract_arguments_from_tuple(
        self,
        ty_kernel_args_tuple: UniTuple,
        ll_kernel_args_tuple: llvmir.Instruction,
    ) -> list[llvmir.Instruction]:
        """Extracts LLVM IR values from llvm tuple into python array."""

        kernel_args = []
        for pos in range(len(ty_kernel_args_tuple)):
            kernel_args.append(
                self.builder.extract_value(ll_kernel_args_tuple, pos)
            )

        return kernel_args

    def allocate_meminfo_array(
        self,
        kernel_argtys: tuple[types.Type, ...],
        kernel_args: [llvmir.Instruction, ...],
    ) -> tuple[int, list[llvmir.Instruction]]:
        """Allocates an LLVM array value to store each memory info from all
        kernel arguments. The array is the populated with the LLVM value for
        every meminfo of the kernel arguments.
        """
        meminfos = []
        for arg_num, argtype in enumerate(kernel_argtys):
            llvm_val = kernel_args[arg_num]

            meminfos += [
                meminfo
                for ty, meminfo in self.context.nrt.get_meminfos(
                    self.builder, argtype, llvm_val
                )
            ]

        meminfo_list = cgutils.alloca_once(
            self.builder,
            utils.get_llvm_type(context=self.context, type=types.voidptr),
            size=self.context.get_constant(types.uintp, len(meminfos)),
        )

        for meminfo_num, meminfo in enumerate(meminfos):
            meminfo_arg_dst = self.builder.gep(
                meminfo_list,
                [self.context.get_constant(types.int32, meminfo_num)],
            )
            meminfo_ptr = self.builder.bitcast(
                meminfo,
                utils.get_llvm_type(context=self.context, type=types.voidptr),
            )
            self.builder.store(meminfo_ptr, meminfo_arg_dst)

        return len(meminfos), meminfo_list

    def get_queue_ref_val(
        self,
        kernel_argtys: tuple[types.Type, ...],
        kernel_args: [llvmir.Instruction, ...],
    ):
        """
        Get the sycl queue from the first DpnpNdArray argument. Prior passes
        before lowering make sure that compute-follows-data is enforceable
        for a specific call to a kernel. As such, at the stage of lowering
        the queue from the first DpnpNdArray argument can be extracted.
        """

        for arg_num, argty in enumerate(kernel_argtys):
            if isinstance(argty, DpnpNdArray):
                llvm_val = kernel_args[arg_num]
                datamodel = self.context.data_model_manager.lookup(argty)
                sycl_queue_attr_pos = datamodel.get_field_position("sycl_queue")
                ptr_to_queue_ref = self.builder.extract_value(
                    llvm_val, sycl_queue_attr_pos
                )
                break

        return ptr_to_queue_ref

    def get_kernel(self, qref, kernel_module: _KernelModule):
        """Returns the pointer to the sycl::kernel object in a passed in
        sycl::kernel_bundle wrapper object.
        """
        # Inserts a global constant byte string in the current LLVM module to
        # store the passed in SPIR-V binary blob.
        kernel_bc_byte_str = self.context.insert_const_bytes(
            self.builder.module,
            bytes=kernel_module.kernel_bitcode,
        )

        # Create a sycl::kernel_bundle object and return it as an opaque pointer
        # using dpctl's libsyclinterface.
        kbref = self.create_kernel_bundle_from_spirv(
            queue_ref=qref,
            kernel_bc=kernel_bc_byte_str,
            kernel_bc_size_in_bytes=len(kernel_module.kernel_bitcode),
        )

        kernel_name = self.context.insert_const_string(
            self.builder.module, kernel_module.kernel_name
        )

        kernel_ref = sycl.dpctl_kernel_bundle_get_kernel(
            self.builder, kbref, kernel_name
        )

        sycl.dpctl_kernel_bundle_delete(self.builder, kbref)

        return kernel_ref

    def create_llvm_values_for_index_space(
        self,
        ty_indexer_arg: Union[RangeType, NdRangeType],
        index_arg: llvmir.BaseStructType,
    ) -> LLRange:
        """Returns two lists of LLVM IR Values that hold the unboxed extents of
        a Python Range or NdRange object.
        """
        ndim = ty_indexer_arg.ndim
        global_range_extents = []
        local_range_extents = []
        indexer_datamodel = numba_default_dmm.lookup(ty_indexer_arg)

        if isinstance(ty_indexer_arg, RangeType):
            for dim_num in range(ndim):
                dim_pos = indexer_datamodel.get_field_position(
                    "dim" + str(dim_num)
                )
                global_range_extents.append(
                    self.builder.extract_value(index_arg, dim_pos)
                )
        elif isinstance(ty_indexer_arg, NdRangeType):
            for dim_num in range(ndim):
                gdim_pos = indexer_datamodel.get_field_position(
                    "gdim" + str(dim_num)
                )
                global_range_extents.append(
                    self.builder.extract_value(index_arg, gdim_pos)
                )
                ldim_pos = indexer_datamodel.get_field_position(
                    "ldim" + str(dim_num)
                )
                local_range_extents.append(
                    self.builder.extract_value(index_arg, ldim_pos)
                )
        else:
            raise UnreachableError

        return LLRange(global_range_extents, local_range_extents)

    def create_kernel_bundle_from_spirv(
        self,
        queue_ref: llvmir.PointerType,
        kernel_bc: llvmir.Constant,
        kernel_bc_size_in_bytes: int,
    ) -> llvmir.CallInstr:
        """Calls DPCTLKernelBundle_CreateFromSpirv to create an opaque pointer
        to a sycl::kernel_bundle from the SPIR-V generated for a kernel.
        """
        device_ref = sycl.dpctl_queue_get_device(self.builder, queue_ref)
        context_ref = sycl.dpctl_queue_get_context(self.builder, queue_ref)
        args = [
            context_ref,
            device_ref,
            kernel_bc,
            llvmir.Constant(llvmir.IntType(64), kernel_bc_size_in_bytes),
            self.builder.load(create_null_ptr(self.builder, self.context)),
        ]
        kb_ref = sycl.dpctl_kernel_bundle_create_from_spirv(self.builder, *args)
        sycl.dpctl_context_delete(self.builder, context_ref)
        sycl.dpctl_device_delete(self.builder, device_ref)

        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(
                self.builder,
                "DPEX-DEBUG: Generated kernel_bundle from SPIR-V.\n",
            )

        return kb_ref

    def acquire_meminfo_and_schedule_release(
        self,
        queue_ref,
        event_ref,
        ty_kernel_args,
        kernel_args,
    ):
        """Schedule sycl host task to release nrt meminfo of the arguments used
        to run job. Use it to keep arguments alive during kernel execution."""
        total_meminfos, meminfo_list = self.allocate_meminfo_array(
            ty_kernel_args, kernel_args
        )

        event_ref_ptr = self.builder.alloca(event_ref.type)
        self.builder.store(event_ref, event_ref_ptr)

        status_ptr = cgutils.alloca_once(
            self.builder, self.context.get_value_type(types.uint64)
        )
        # TODO: get dpex RT from cached property once the PR is merged
        # https://github.com/IntelPython/numba-dpex/pull/1027
        # host_eref = ctx.dpexrt.acquire_meminfo_and_schedule_release( # noqa: W0621
        host_eref = DpexRTContext(
            self.context
        ).acquire_meminfo_and_schedule_release(
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


@intrinsic(target="cpu")
def _submit_kernel(
    typingctx,  # pylint: disable=W0613
    ty_kernel_fn: Dispatcher,
    ty_index_space: Union[RangeType, NdRangeType],
    ty_kernel_args_tuple: UniTuple,
):
    """Generates IR code for call_kernel dpjit function.

    The intrinsic first compiles the kernel function to SPIRV, and then to a
    sycl kernel bundle. The arguments to the kernel are also packed into
    flattened arrays and the sycl queue to which the kernel will be submitted
    extracted from the args. Finally, the actual kernel is extracted from the
    kernel bundle and submitted to the sycl queue.
    """
    # signature of this intrinsic
    ty_event = DpctlSyclEvent()
    sig = ty_event(ty_kernel_fn, ty_index_space, ty_kernel_args_tuple)
    kernel_sig = types.void(*ty_kernel_args_tuple)
    # ty_kernel_fn is type specific to exact function, so we can get function
    # directly from type and compile it. Thats why we don't need to get it in
    # codegen
    kernel_dispatcher: KernelDispatcher = ty_kernel_fn.dispatcher
    kernel_dispatcher.compile(kernel_sig)
    kernel_module: _KernelModule = kernel_dispatcher.get_overload_device_ir(
        kernel_sig
    )
    kernel_targetctx = kernel_dispatcher.targetctx

    def codegen(cgctx, builder, sig, llargs):
        # llargs[0] is kernel function that we don't need anymore (see above)
        ty_index_space: Union[RangeType, NdRangeType] = sig.args[1]
        ll_index_space: llvmir.Instruction = llargs[1]
        ty_kernel_args_tuple: UniTuple = sig.args[2]
        ll_kernel_args_tuple: llvmir.Instruction = llargs[2]

        generator = CallKernelBuilder(
            codegen_targetctx=cgctx,
            builder=builder,
        )

        kernel_args = generator.extract_arguments_from_tuple(
            ty_kernel_args_tuple=ty_kernel_args_tuple,
            ll_kernel_args_tuple=ll_kernel_args_tuple,
        )

        # queue_ref is just a pointer to the attribute, so we don't have to
        # clean it up
        queue_ref = generator.get_queue_ref_val(
            kernel_argtys=ty_kernel_args_tuple,
            kernel_args=kernel_args,
        )

        # creates new object, so we must clean it up
        kernel_ref = generator.get_kernel(queue_ref, kernel_module)

        ll_range = generator.create_llvm_values_for_index_space(
            ty_indexer_arg=ty_index_space,
            index_arg=ll_index_space,
        )

        device_event_ref = kl.KernelLaunchIRBuilder(
            kernel_targetctx, builder
        ).submit_kernel(
            kernel_ref=kernel_ref,
            queue_ref=queue_ref,
            kernel_args=kernel_args,
            ty_kernel_args=ty_kernel_args_tuple,
            global_range_extents=ll_range.global_range_extents,
            local_range_extents=ll_range.local_range_extents,
        )

        # Clean up
        sycl.dpctl_kernel_delete(builder, kernel_ref)

        return wrap_event_reference(cgctx, builder, device_event_ref)

    return sig, codegen


@intrinsic(target="cpu")
def _acquire_meminfo_and_schedule_release(
    typingctx,  # pylint: disable=W0613
    ty_device_event,  # pylint: disable=W0613
    ty_kernel_args,
):
    """Generates IR code to keep arguments alive during kernel execution.

    The intrinsic collects all memory infos from the kernel arguments, acquires
    them and schecules host task to release them. Returns host task's event.
    """
    # signature of this intrinsic
    ty_event = DpctlSyclEvent()
    sig = ty_event(ty_event, ty_kernel_args)

    def codegen(cgctx, builder, sig, llargs):
        device_event = cgutils.create_struct_proxy(sig.args[0])(
            cgctx, builder, value=llargs[0]
        )

        ll_kernel_args_tuple = llargs[1]
        ty_kernel_args_tuple = sig.args[1]

        generator = CallKernelBuilder(
            codegen_targetctx=cgctx,
            builder=builder,
        )

        kernel_args = generator.extract_arguments_from_tuple(
            ty_kernel_args_tuple=ty_kernel_args_tuple,
            ll_kernel_args_tuple=ll_kernel_args_tuple,
        )

        qref = generator.get_queue_ref_val(
            kernel_argtys=ty_kernel_args_tuple,
            kernel_args=kernel_args,
        )

        host_eref = generator.acquire_meminfo_and_schedule_release(
            queue_ref=qref,
            event_ref=device_event.event_ref,
            ty_kernel_args=ty_kernel_args_tuple,
            kernel_args=kernel_args,
        )

        return wrap_event_reference(cgctx, builder, host_eref)

    return sig, codegen


@dpjit
def call_kernel(kernel_fn, index_space, *kernel_args) -> None:
    """Calls a numba_dpex.kernel decorated function from CPython or from another
    dpjit function. Kernel execution happens in syncronous way, so the thread
    will be blocked till the kernel done exectuion.

    Args:
        kernel_fn (numba_dpex.experimental.KernelDispatcher): A
        numba_dpex.kernel decorated function that is compiled to a
        KernelDispatcher by numba_dpex.
        index_space (Range | NdRange): A numba_dpex.Range or numba_dpex.NdRange
        type object that specifies the index space for the kernel.
        kernel_args : List of objects that are passed to the numba_dpex.kernel
        decorated function.
    """
    device_event = _submit_kernel(  # pylint: disable=E1120
        kernel_fn,
        index_space,
        kernel_args,
    )
    device_event.wait()  # pylint: disable=E1101


@dpjit
def call_kernel_async(
    kernel_fn, index_space, *kernel_args
) -> tuple[dpctl.SyclEvent, dpctl.SyclEvent]:
    """Calls a numba_dpex.kernel decorated function from CPython or from another
    dpjit function. Kernel execution happens in asyncronous way, so the thread
    will not be blocked till the kernel done exectuion. That means that it is
    user responsiblity to properly use any memory used by kernel until the
    kernel execution is completed.

    Args:
        kernel_fn (numba_dpex.experimental.KernelDispatcher): A
        numba_dpex.kernel decorated function that is compiled to a
        KernelDispatcher by numba_dpex.
        index_space (Range | NdRange): A numba_dpex.Range or numba_dpex.NdRange
        type object that specifies the index space for the kernel.
        kernel_args : List of objects that are passed to the numba_dpex.kernel
        decorated function.

    Returns:
        pair of host event and device event. Host event represent host task
        that releases use of any kernel argument so it can be deallocated.
        This task may be executed only after device task is done.
    """
    device_event = _submit_kernel(  # pylint: disable=E1120
        kernel_fn,
        index_space,
        kernel_args,
    )
    host_event = _acquire_meminfo_and_schedule_release(  # pylint: disable=E1120
        device_event, kernel_args
    )
    return host_event, device_event
