# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides a helper function to call a numba_dpex.kernel decorated function
from either CPython or a numba_dpex.dpjit decorated function.
"""

from collections import namedtuple
from typing import Union

from llvmlite import ir as llvmir
from numba.core import cgutils, cpu, types
from numba.extending import intrinsic, overload

from numba_dpex import config, dpjit
from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.targets.kernel_target import DpexKernelTargetContext
from numba_dpex.core.types import DpnpNdArray, NdRangeType, RangeType
from numba_dpex.core.utils import kernel_launcher as kl
from numba_dpex.dpctl_iface import libsyclinterface_bindings as sycl
from numba_dpex.experimental.kernel_dispatcher import _KernelModule
from numba_dpex.utils import create_null_ptr

_KernelSubmissionArgs = namedtuple(
    "_KernelSubmissionArgs",
    [
        "kernel_ref",
        "queue_ref",
        "flattened_args_count",
        "array_of_kernel_args",
        "array_of_kernel_arg_types",
        "global_range_extents",
        "local_range_extents",
    ],
)

_LLVMIRValuesForIndexSpace = namedtuple(
    "_LLVMIRValuesForNdRange", ["global_range_extents", "local_range_extents"]
)


def _get_queue_ref_val(
    kernel_targetctx: DpexKernelTargetContext,
    builder: llvmir.IRBuilder,
    kernel_argtys: [types.Type, ...],
    kernel_args,
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
            datamodel = kernel_targetctx.data_model_manager.lookup(argty)
            sycl_queue_attr_pos = datamodel.get_field_position("sycl_queue")
            ptr_to_queue_ref = builder.extract_value(
                llvm_val, sycl_queue_attr_pos
            )
            break

    return ptr_to_queue_ref


def _create_kernel_bundle_from_spirv(
    builder: llvmir.IRBuilder,
    targetctx: cpu.CPUContext,
    queue_ref: llvmir.PointerType,
    kernel_bc: llvmir.Constant,
    kernel_bc_size_in_bytes: int,
):
    dref = sycl.dpctl_queue_get_device(builder, queue_ref)
    cref = sycl.dpctl_queue_get_context(builder, queue_ref)
    args = [
        cref,
        dref,
        kernel_bc,
        llvmir.Constant(llvmir.IntType(64), kernel_bc_size_in_bytes),
        builder.load(create_null_ptr(builder, targetctx)),
    ]
    kbref = sycl.dpctl_kernel_bundle_create_from_spirv(builder, *args)
    sycl.dpctl_context_delete(builder, cref)
    sycl.dpctl_device_delete(builder, dref)

    return kbref


def _get_num_flattened_kernel_args(
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


def _create_llvm_values_for_index_space(
    builder: llvmir.IRBuilder,
    kernel_targetctx: DpexKernelTargetContext,
    indexer_argty: NdRangeType,
    index_space_arg: llvmir.BaseStructType,
) -> _LLVMIRValuesForIndexSpace:
    """Returns a list of LLVM IR Values that hold the unboxed extents of a
    Python Range or NdRange object.
    """
    ndim = indexer_argty.ndim
    grange_extents = []
    lrange_extents = []
    datamodel = kernel_targetctx.data_model_manager.lookup(indexer_argty)

    if isinstance(indexer_argty, RangeType):
        for dim_num in range(ndim):
            dim_pos = datamodel.get_field_position("dim" + str(dim_num))
            grange_extents.append(
                builder.extract_value(index_space_arg, dim_pos)
            )
    elif isinstance(indexer_argty, NdRangeType):
        for dim_num in range(ndim):
            gdim_pos = datamodel.get_field_position("gdim" + str(dim_num))
            grange_extents.append(
                builder.extract_value(index_space_arg, gdim_pos)
            )
            ldim_pos = datamodel.get_field_position("ldim" + str(dim_num))
            lrange_extents.append(
                builder.extract_value(index_space_arg, ldim_pos)
            )
    else:
        raise UnreachableError

    return _LLVMIRValuesForIndexSpace(
        global_range_extents=grange_extents, local_range_extents=lrange_extents
    )


def _create_kernel_launcher_body(
    codegen_targetctx: cpu.CPUContext,
    kernel_targetctx: DpexKernelTargetContext,
    builder: llvmir.IRBuilder,
    indexer_argty: Union[RangeType, NdRangeType],
    kernel_argtys: tuple[types.Type, ...],
    kernel_module: _KernelModule,
    index_space_arg: llvmir.BaseStructType,
    kernel_args: [llvmir.Instruction, ...],
):
    klbuilder = kl.KernelLaunchIRBuilder(kernel_targetctx, builder)

    if config.DEBUG_KERNEL_LAUNCHER:
        cgutils.printf(
            builder, "DPEX-DEBUG: Inside the kernel launcher function\n"
        )

    kernel_bc_byte_str: llvmir.Constant = codegen_targetctx.insert_const_bytes(
        builder.module,
        bytes=kernel_module.kernel_bitcode,
    )

    num_flattened_kernel_args = _get_num_flattened_kernel_args(
        kernel_targetctx=kernel_targetctx, kernel_argtys=kernel_argtys
    )

    # Create LLVM values for the kernel args list and kernel arg types list
    args_list = klbuilder.allocate_kernel_arg_array(num_flattened_kernel_args)
    args_ty_list = klbuilder.allocate_kernel_arg_ty_array(
        num_flattened_kernel_args
    )
    kernel_args_ptrs = []
    for arg in kernel_args:
        ptr = builder.alloca(arg.type)
        builder.store(arg, ptr)
        kernel_args_ptrs.append(ptr)

    # Populate the args_list and the args_ty_list LLVM arrays
    klbuilder.populate_kernel_args_and_args_ty_arrays(
        callargs_ptrs=kernel_args_ptrs,
        kernel_argtys=kernel_argtys,
        args_list=args_list,
        args_ty_list=args_ty_list,
        datamodel_mgr=kernel_targetctx.data_model_manager,
    )

    if config.DEBUG_KERNEL_LAUNCHER:
        cgutils.printf(
            builder, "DPEX-DEBUG: Populated kernel args and arg type arrays.\n"
        )

    qref = _get_queue_ref_val(
        kernel_targetctx=kernel_targetctx,
        builder=builder,
        kernel_argtys=kernel_argtys,
        kernel_args=kernel_args,
    )

    if config.DEBUG_KERNEL_LAUNCHER:
        cgutils.printf(
            builder,
            "DPEX-DEBUG: Extracted queue pointer from first dpnp array.\n",
        )

    kbref = _create_kernel_bundle_from_spirv(
        builder=builder,
        targetctx=codegen_targetctx,
        queue_ref=qref,
        kernel_bc=kernel_bc_byte_str,
        kernel_bc_size_in_bytes=len(kernel_module.kernel_bitcode),
    )

    if config.DEBUG_KERNEL_LAUNCHER:
        cgutils.printf(
            builder, "DPEX-DEBUG: Generated kernel_bundle from SPIR-V.\n"
        )

    # Get the pointer to the sycl::kernel object in the sycl::kernel_bundle
    kernel_name = codegen_targetctx.insert_const_string(
        builder.module, kernel_module.kernel_name
    )
    kref = sycl.dpctl_kernel_bundle_get_kernel(builder, kbref, kernel_name)

    # Submit synchronous kernel
    # TODO: Needs to change once we support returning a SyclEvent back to
    # caller.

    index_space_values = _create_llvm_values_for_index_space(
        builder=builder,
        kernel_targetctx=kernel_targetctx,
        indexer_argty=indexer_argty,
        index_space_arg=index_space_arg,
    )

    submit_call_args = _KernelSubmissionArgs(
        kernel_ref=kref,
        queue_ref=qref,
        flattened_args_count=num_flattened_kernel_args,
        array_of_kernel_args=args_list,
        array_of_kernel_arg_types=args_ty_list,
        global_range_extents=index_space_values.global_range_extents,
        local_range_extents=index_space_values.local_range_extents,
    )

    if config.DEBUG_KERNEL_LAUNCHER:
        cgutils.printf(builder, "DPEX-DEBUG: Submit sync range kernel.\n")

    eref = klbuilder.submit_sycl_kernel(
        sycl_kernel_ref=submit_call_args.kernel_ref,
        sycl_queue_ref=submit_call_args.queue_ref,
        total_kernel_args=submit_call_args.flattened_args_count,
        arg_list=submit_call_args.array_of_kernel_args,
        arg_ty_list=submit_call_args.array_of_kernel_arg_types,
        global_range=submit_call_args.global_range_extents,
        local_range=submit_call_args.local_range_extents,
        wait_before_return=False,
    )
    if config.DEBUG_KERNEL_LAUNCHER:
        cgutils.printf(builder, "DPEX-DEBUG: Wait on event.\n")

    sycl.dpctl_event_wait(builder, eref)
    sycl.dpctl_event_delete(builder, eref)
    # Delete the kernel ref
    sycl.dpctl_kernel_delete(builder, kref)
    # Delete the kernel bundle pointer
    sycl.dpctl_kernel_bundle_delete(builder, kbref)


@intrinsic(target="cpu")
def intrin_launch_trampoline(
    typingctx, kernel_fn, index_space, kernel_args  # pylint: disable=W0613
):
    kernel_args_list = list(kernel_args)
    # signature of this intrinsic
    sig = types.void(kernel_fn, index_space, kernel_args)
    # signature of the kernel_fn
    kernel_sig = types.void(*kernel_args_list)
    kmodule: _KernelModule = kernel_fn.dispatcher.compile(kernel_sig)
    kernel_targetctx = kernel_fn.dispatcher.targetctx

    def codegen(cgctx, builder, sig, llargs):
        kernel_argtys = kernel_sig.args
        kernel_args_unpacked = []
        for pos in range(len(kernel_args)):
            kernel_args_unpacked.append(builder.extract_value(llargs[2], pos))
        _create_kernel_launcher_body(
            codegen_targetctx=cgctx,
            kernel_targetctx=kernel_targetctx,
            builder=builder,
            indexer_argty=sig.args[1],
            kernel_argtys=kernel_argtys,
            kernel_module=kmodule,
            index_space_arg=llargs[1],
            kernel_args=kernel_args_unpacked,
        )

    return sig, codegen


# pylint: disable=W0613
def _launch_trampoline(kernel_fn, index_space, *kernel_args):
    pass


@overload(_launch_trampoline, target="cpu")
def _ol_launch_trampoline(kernel_fn, index_space, *kernel_args):
    def impl(kernel_fn, index_space, *kernel_args):
        intrin_launch_trampoline(  # pylint: disable=E1120
            kernel_fn, index_space, kernel_args
        )

    return impl


@dpjit
def call_kernel(kernel_fn, index_space, *kernel_args):
    """Calls a numba_dpex.kernel decorated function from CPython or from another
    dpjit function.

    Args:
        kernel_fn (numba_dpex.experimental.KernelDispatcher): A
        numba_dpex.kernel decorated function that is compiled to a
        KernelDispatcher by numba_dpex.
        index_space (Range | NdRange): A numba_dpex.Range or numba_dpex.NdRange
        type object that specifies the index space for the kernel.
        kernel_args : List of objects that are passed to the numba_dpex.kernel
        decorated function.
    """
    _launch_trampoline(kernel_fn, index_space, *kernel_args)
