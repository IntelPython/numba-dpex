# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides a helper function to call a numba_dpex.kernel decorated function
from either CPython or a numba_dpex.dpjit decorated function.
"""

from collections import namedtuple
from typing import Union

import dpctl
from llvmlite import ir as llvmir
from numba.core import cgutils, cpu, types
from numba.core.datamodel import default_manager as numba_default_dmm
from numba.extending import intrinsic

from numba_dpex import config, dpjit, utils
from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.runtime.context import DpexRTContext
from numba_dpex.core.targets.kernel_target import DpexKernelTargetContext
from numba_dpex.core.types import (
    DpctlSyclEvent,
    DpnpNdArray,
    NdRangeType,
    RangeType,
)
from numba_dpex.core.utils import kernel_launcher as kl
from numba_dpex.dpctl_iface import libsyclinterface_bindings as sycl
from numba_dpex.dpctl_iface.wrappers import wrap_event_reference
from numba_dpex.experimental.kernel_dispatcher import _KernelModule
from numba_dpex.utils import create_null_ptr

_KernelArgs = namedtuple(
    "_KernelArgs",
    [
        "flattened_args_count",
        "array_of_kernel_args",
        "array_of_kernel_arg_types",
    ],
)

_KernelSubmissionArgs = namedtuple(
    "_KernelSubmissionArgs",
    [
        "kernel_ref",
        "queue_ref",
        "kernel_args",
        "global_range_extents",
        "local_range_extents",
    ],
)

_LLVMIRValuesForIndexSpace = namedtuple(
    "_LLVMIRValuesForNdRange", ["global_range_extents", "local_range_extents"]
)


class _LaunchTrampolineFunctionBodyGenerator:
    """
    Helper class to generate the LLVM IR for the launch_trampoline intrinsic.
    """

    def _get_num_flattened_kernel_args(
        self,
        kernel_targetctx: DpexKernelTargetContext,
        kernel_argtys: tuple[types.Type, ...],
    ):
        num_flattened_kernel_args = 0
        for arg_type in kernel_argtys:
            if isinstance(arg_type, DpnpNdArray):
                datamodel = kernel_targetctx.data_model_manager.lookup(arg_type)
                num_flattened_kernel_args += datamodel.flattened_field_count
            elif arg_type in [types.complex64, types.complex128]:
                num_flattened_kernel_args += 2
            else:
                num_flattened_kernel_args += 1

        return num_flattened_kernel_args

    def __init__(
        self,
        codegen_targetctx: cpu.CPUContext,
        kernel_targetctx: DpexKernelTargetContext,
        builder: llvmir.IRBuilder,
    ):
        self._cpu_codegen_targetctx = codegen_targetctx
        self._kernel_targetctx = kernel_targetctx
        self._builder = builder
        if kernel_targetctx:
            self._klbuilder = kl.KernelLaunchIRBuilder(
                kernel_targetctx, builder
            )

        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(
                self._builder,
                "DPEX-DEBUG: Inside the kernel launcher function\n",
            )

    def insert_kernel_bitcode_as_byte_str(
        self, kernel_module: _KernelModule
    ) -> None:
        """Inserts a global constant byte string in the current LLVM module to
        store the passed in SPIR-V binary blob.
        """
        return self._cpu_codegen_targetctx.insert_const_bytes(
            self._builder.module,
            bytes=kernel_module.kernel_bitcode,
        )

    def allocate_meminfos_array(self, num_meminfos):
        """Allocates an array to store nrt memory infos.

        Args:
            num_meminfos (int): The number of memory infos to allocate.

        Returns: An LLVM IR value pointing to an array to store the memory
        infos.
        """
        builder = self._builder
        context = self._cpu_codegen_targetctx

        meminfo_list = cgutils.alloca_once(
            builder,
            utils.get_llvm_type(context=context, type=types.voidptr),
            size=context.get_constant(types.uintp, num_meminfos),
        )

        return meminfo_list

    def populate_kernel_args_and_argsty_arrays(
        self,
        kernel_argtys: tuple[types.Type, ...],
        kernel_args: [llvmir.Instruction, ...],
    ) -> _KernelArgs:
        """Allocates an LLVM array value to store each flattened kernel arg and
        another LLVM array to store the typeid for each flattened kernel arg.
        The arrays are the populated with the LLVM value for each arg.
        """
        num_flattened_kernel_args = self._get_num_flattened_kernel_args(
            kernel_targetctx=self._kernel_targetctx, kernel_argtys=kernel_argtys
        )

        # Create LLVM values for the kernel args list and kernel arg types list
        args_list = self._klbuilder.allocate_kernel_arg_array(
            num_flattened_kernel_args
        )
        args_ty_list = self._klbuilder.allocate_kernel_arg_ty_array(
            num_flattened_kernel_args
        )
        kernel_args_ptrs = []
        for arg in kernel_args:
            ptr = self._builder.alloca(arg.type)
            self._builder.store(arg, ptr)
            kernel_args_ptrs.append(ptr)

        # Populate the args_list and the args_ty_list LLVM arrays
        self._klbuilder.populate_kernel_args_and_args_ty_arrays(
            callargs_ptrs=kernel_args_ptrs,
            kernel_argtys=kernel_argtys,
            args_list=args_list,
            args_ty_list=args_ty_list,
            datamodel_mgr=self._kernel_targetctx.data_model_manager,
        )

        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(
                self._builder,
                "DPEX-DEBUG: Populated kernel args and arg type arrays.\n",
            )

        return _KernelArgs(
            flattened_args_count=num_flattened_kernel_args,
            array_of_kernel_args=args_list,
            array_of_kernel_arg_types=args_ty_list,
        )

    def allocate_meminfo_array(
        self,
        kernel_argtys: tuple[types.Type, ...],
        kernel_args: [llvmir.Instruction, ...],
    ) -> tuple[int, list[llvmir.Instruction]]:
        """Allocates an LLVM array value to store each memory info from all
        kernel arguments. The array is the populated with the LLVM value for
        every meminfo of the kernel arguments.
        """
        builder = self._builder
        context = self._cpu_codegen_targetctx

        meminfos = []
        for arg_num, argtype in enumerate(kernel_argtys):
            llvm_val = kernel_args[arg_num]

            meminfos += [
                meminfo
                for ty, meminfo in context.nrt.get_meminfos(
                    builder, argtype, llvm_val
                )
            ]

        meminfo_list = self.allocate_meminfos_array(len(meminfos))

        for meminfo_num, meminfo in enumerate(meminfos):
            meminfo_arg_dst = builder.gep(
                meminfo_list,
                [context.get_constant(types.int32, meminfo_num)],
            )
            meminfo_ptr = builder.bitcast(
                meminfo,
                utils.get_llvm_type(context=context, type=types.voidptr),
            )
            builder.store(meminfo_ptr, meminfo_arg_dst)

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
                datamodel = (
                    self._cpu_codegen_targetctx.data_model_manager.lookup(argty)
                )
                sycl_queue_attr_pos = datamodel.get_field_position("sycl_queue")
                ptr_to_queue_ref = self._builder.extract_value(
                    llvm_val, sycl_queue_attr_pos
                )
                break

        return ptr_to_queue_ref

    def get_kernel(self, kernel_module, kbref):
        """Returns the pointer to the sycl::kernel object in a passed in
        sycl::kernel_bundle wrapper object.
        """
        kernel_name = self._cpu_codegen_targetctx.insert_const_string(
            self._builder.module, kernel_module.kernel_name
        )
        return sycl.dpctl_kernel_bundle_get_kernel(
            self._builder, kbref, kernel_name
        )

    def create_llvm_values_for_index_space(
        self,
        indexer_argty: Union[RangeType, NdRangeType],
        index_space_arg: llvmir.BaseStructType,
    ) -> _LLVMIRValuesForIndexSpace:
        """Returns a list of LLVM IR Values that hold the unboxed extents of a
        Python Range or NdRange object.
        """
        ndim = indexer_argty.ndim
        grange_extents = []
        lrange_extents = []
        indexer_datamodel = numba_default_dmm.lookup(indexer_argty)

        if isinstance(indexer_argty, RangeType):
            for dim_num in range(ndim):
                dim_pos = indexer_datamodel.get_field_position(
                    "dim" + str(dim_num)
                )
                grange_extents.append(
                    self._builder.extract_value(index_space_arg, dim_pos)
                )
        elif isinstance(indexer_argty, NdRangeType):
            for dim_num in range(ndim):
                gdim_pos = indexer_datamodel.get_field_position(
                    "gdim" + str(dim_num)
                )
                grange_extents.append(
                    self._builder.extract_value(index_space_arg, gdim_pos)
                )
                ldim_pos = indexer_datamodel.get_field_position(
                    "ldim" + str(dim_num)
                )
                lrange_extents.append(
                    self._builder.extract_value(index_space_arg, ldim_pos)
                )
        else:
            raise UnreachableError

        return _LLVMIRValuesForIndexSpace(
            global_range_extents=grange_extents,
            local_range_extents=lrange_extents,
        )

    def create_kernel_bundle_from_spirv(
        self,
        queue_ref: llvmir.PointerType,
        kernel_bc: llvmir.Constant,
        kernel_bc_size_in_bytes: int,
    ) -> llvmir.CallInstr:
        """Calls DPCTLKernelBundle_CreateFromSpirv to create an opaque pointer
        to a sycl::kernel_bundle from the SPIR-V generated for a kernel.
        """
        dref = sycl.dpctl_queue_get_device(self._builder, queue_ref)
        cref = sycl.dpctl_queue_get_context(self._builder, queue_ref)
        args = [
            cref,
            dref,
            kernel_bc,
            llvmir.Constant(llvmir.IntType(64), kernel_bc_size_in_bytes),
            self._builder.load(
                create_null_ptr(self._builder, self._cpu_codegen_targetctx)
            ),
        ]
        kbref = sycl.dpctl_kernel_bundle_create_from_spirv(self._builder, *args)
        sycl.dpctl_context_delete(self._builder, cref)
        sycl.dpctl_device_delete(self._builder, dref)

        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(
                self._builder,
                "DPEX-DEBUG: Generated kernel_bundle from SPIR-V.\n",
            )

        return kbref

    def submit(
        self, submit_call_args: _KernelSubmissionArgs
    ) -> llvmir.PointerType(llvmir.IntType(8)):
        """Generates LLVM IR CallInst to submit a kernel to specified SYCL
        queue.
        """
        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(
                self._builder, "DPEX-DEBUG: Submit sync range kernel.\n"
            )

        eref = self._klbuilder.submit_sycl_kernel(
            sycl_kernel_ref=submit_call_args.kernel_ref,
            sycl_queue_ref=submit_call_args.queue_ref,
            total_kernel_args=submit_call_args.kernel_args.flattened_args_count,
            arg_list=submit_call_args.kernel_args.array_of_kernel_args,
            arg_ty_list=submit_call_args.kernel_args.array_of_kernel_arg_types,
            global_range=submit_call_args.global_range_extents,
            local_range=submit_call_args.local_range_extents,
            wait_before_return=False,
        )
        if config.DEBUG_KERNEL_LAUNCHER:
            cgutils.printf(self._builder, "DPEX-DEBUG: Wait on event.\n")

        return eref

    def acquire_meminfo_and_schedule_release(
        self,
        qref,
        eref,
        total_meminfos,
        meminfo_list,
    ):
        """Schedule sycl host task to release nrt meminfo of the arguments used
        to run job. Use it to keep arguments alive during kernel execution."""
        ctx = self._cpu_codegen_targetctx
        builder = self._builder

        eref_ptr = builder.alloca(eref.type)
        builder.store(eref, eref_ptr)

        status_ptr = cgutils.alloca_once(
            builder, ctx.get_value_type(types.uint64)
        )
        # TODO: get dpex RT from cached property once the PR is merged
        # https://github.com/IntelPython/numba-dpex/pull/1027
        # host_eref = ctx.dpexrt.acquire_meminfo_and_schedule_release( # noqa: W0621
        host_eref = DpexRTContext(ctx).acquire_meminfo_and_schedule_release(
            builder,
            [
                ctx.nrt.get_nrt_api(builder),
                qref,
                meminfo_list,
                ctx.get_constant(types.uintp, total_meminfos),
                eref_ptr,
                ctx.get_constant(types.uintp, 1),
                status_ptr,
            ],
        )
        return host_eref

    def cleanup(
        self,
        kernel_ref: llvmir.Instruction,
        kernel_bundle_ref: llvmir.Instruction,
    ) -> None:
        """Generates calls to free up temporary resources that were allocated in
        the launch_trampoline body.
        """
        # Delete the kernel ref
        sycl.dpctl_kernel_delete(self._builder, kernel_ref)
        # Delete the kernel bundle pointer
        sycl.dpctl_kernel_bundle_delete(self._builder, kernel_bundle_ref)


@intrinsic(target="cpu")
def _submit_kernel(
    typingctx,  # pylint: disable=W0613
    kernel_fn,
    index_space,
    kernel_args,
):
    """Generates IR code for call_kernel dpjit function.

    The intrinsic first compiles the kernel function to SPIRV, and then to a
    sycl kernel bundle. The arguments to the kernel are also packed into
    flattened arrays and the sycl queue to which the kernel will be submitted
    extracted from the args. Finally, the actual kernel is extracted from the
    kernel bundle and submitted to the sycl queue.
    """
    kernel_args_list = list(kernel_args)
    # signature of this intrinsic
    ty_event = DpctlSyclEvent()
    sig = ty_event(kernel_fn, index_space, kernel_args)
    # signature of the kernel_fn
    kernel_sig = types.void(*kernel_args_list)
    kernel_fn.dispatcher.compile(kernel_sig)
    kernel_module: _KernelModule = kernel_fn.dispatcher.get_overload_device_ir(
        kernel_sig
    )
    kernel_targetctx = kernel_fn.dispatcher.targetctx

    # TODO: refactor so there are no too many locals
    def codegen(cgctx, builder, sig, llargs):  # pylint: disable=R0914
        kernel_argtys = kernel_sig.args
        kernel_args_unpacked = []
        for pos in range(len(kernel_args)):
            kernel_args_unpacked.append(builder.extract_value(llargs[2], pos))

        fn_body_gen = _LaunchTrampolineFunctionBodyGenerator(
            codegen_targetctx=cgctx,
            kernel_targetctx=kernel_targetctx,
            builder=builder,
        )

        kernel_bc_byte_str = fn_body_gen.insert_kernel_bitcode_as_byte_str(
            kernel_module
        )

        populated_kernel_args = (
            fn_body_gen.populate_kernel_args_and_argsty_arrays(
                kernel_argtys, kernel_args_unpacked
            )
        )

        qref = fn_body_gen.get_queue_ref_val(
            kernel_argtys=kernel_args_list,
            kernel_args=kernel_args_unpacked,
        )

        kbref = fn_body_gen.create_kernel_bundle_from_spirv(
            queue_ref=qref,
            kernel_bc=kernel_bc_byte_str,
            kernel_bc_size_in_bytes=len(kernel_module.kernel_bitcode),
        )

        kref = fn_body_gen.get_kernel(kernel_module, kbref)

        index_space_values = fn_body_gen.create_llvm_values_for_index_space(
            indexer_argty=sig.args[1],
            index_space_arg=llargs[1],
        )

        submit_call_args = _KernelSubmissionArgs(
            kernel_ref=kref,
            queue_ref=qref,
            kernel_args=populated_kernel_args,
            global_range_extents=index_space_values.global_range_extents,
            local_range_extents=index_space_values.local_range_extents,
        )

        eref = fn_body_gen.submit(submit_call_args)
        # We could've just wait and delete event here, but we want to reuse
        # this function in async kernel submition and unfortunately numba does
        # not support conditional returns:
        # https://github.com/numba/numba/issues/9314
        device_event = wrap_event_reference(cgctx, builder, eref)
        return device_event

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

        kernel_args_tuple = llargs[1]
        ty_kernel_args = sig.args[1]

        kernel_args = []
        for pos in range(len(ty_kernel_args)):
            kernel_args.append(builder.extract_value(kernel_args_tuple, pos))

        fn_body_gen = _LaunchTrampolineFunctionBodyGenerator(
            codegen_targetctx=cgctx,
            kernel_targetctx=None,
            builder=builder,
        )

        total_meminfos, meminfo_list = fn_body_gen.allocate_meminfo_array(
            ty_kernel_args, kernel_args
        )

        qref = fn_body_gen.get_queue_ref_val(
            kernel_argtys=ty_kernel_args,
            kernel_args=kernel_args,
        )

        host_eref = fn_body_gen.acquire_meminfo_and_schedule_release(
            qref=qref,
            eref=device_event.event_ref,
            total_meminfos=total_meminfos,
            meminfo_list=meminfo_list,
        )

        host_event = wrap_event_reference(cgctx, builder, host_eref)

        return host_event

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
