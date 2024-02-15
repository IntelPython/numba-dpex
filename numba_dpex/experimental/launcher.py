# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides a helper function to call a numba_dpex.kernel decorated function
from either CPython or a numba_dpex.dpjit decorated function.
"""

import warnings
from inspect import signature
from typing import NamedTuple, Union

import dpctl
from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.core.cpu import CPUContext
from numba.core.types.containers import Tuple, UniTuple
from numba.core.types.functions import Dispatcher
from numba.extending import intrinsic

from numba_dpex import dpjit
from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME
from numba_dpex.core.types import DpctlSyclEvent, NdRangeType, RangeType
from numba_dpex.core.utils import kernel_launcher as kl
from numba_dpex.dpctl_iface import libsyclinterface_bindings as sycl
from numba_dpex.dpctl_iface.wrappers import wrap_event_reference
from numba_dpex.experimental.core.types.kernel_api.items import (
    ItemType,
    NdItemType,
)
from numba_dpex.kernel_api_impl.spirv.dispatcher import (
    SPIRVKernelDispatcher,
    _SPIRVKernelCompileResult,
)
from numba_dpex.kernel_api_impl.spirv.target import SPIRVTargetContext


class LLRange(NamedTuple):
    """Analog of Range and NdRange but for the llvm ir values."""

    global_range_extents: list
    local_range_extents: list


def wrap_event_reference_tuple(ctx, builder, event1, event2):
    """Creates tuple data model from two event data models, so it can be
    boxed to Python."""
    ty_event = DpctlSyclEvent()
    tupty = types.Tuple([ty_event, ty_event])

    lltupty = ctx.get_value_type(tupty)
    tup = cgutils.get_null_value(lltupty)
    tup = builder.insert_value(tup, event1, 0)
    tup = builder.insert_value(tup, event2, 1)

    return tup


@intrinsic(target=DPEX_TARGET_NAME)
def _submit_kernel_async(
    typingctx,
    ty_kernel_fn: Dispatcher,
    ty_index_space: Union[RangeType, NdRangeType],
    ty_dependent_events: UniTuple,
    ty_kernel_args_tuple: UniTuple,
):
    """Generates IR code for call_kernel_async ``dpjit`` function."""
    return _submit_kernel(
        typingctx,
        ty_kernel_fn,
        ty_index_space,
        ty_dependent_events,
        ty_kernel_args_tuple,
        sync=False,
    )


@intrinsic(target=DPEX_TARGET_NAME)
def _submit_kernel_sync(
    typingctx,
    ty_kernel_fn: Dispatcher,
    ty_index_space: Union[RangeType, NdRangeType],
    ty_kernel_args_tuple: UniTuple,
):
    """Generates IR code for call_kernel ``dpjit`` function."""
    return _submit_kernel(
        typingctx,
        ty_kernel_fn,
        ty_index_space,
        None,
        ty_kernel_args_tuple,
        sync=True,
    )


def _submit_kernel(  # pylint: disable=too-many-arguments
    typingctx,  # pylint: disable=unused-argument
    ty_kernel_fn: Dispatcher,
    ty_index_space: Union[RangeType, NdRangeType],
    ty_dependent_events: UniTuple,
    ty_kernel_args_tuple: UniTuple,
    sync: bool,
):
    """Generates IR code for call_kernel_{async|sync} ``dpjit`` function.

    The intrinsic first compiles the kernel function to SPIR-V, and then to a
    SYCL kernel bundle. The arguments to the kernel are also packed into
    flattened arrays and the SYCL queue to which the kernel will be submitted
    extracted from the args. Finally, the actual kernel is extracted from the
    kernel bundle and submitted to the SYCL queue.

    If sync set to False, it acquires memory infos from kernel arguments to
    prevent garbage collection on them. Then it schedules host task to release
    that arguments and unblock garbage collection. Tuple of host task and device
    tasks are returned.
    """
    # signature of this intrinsic
    ty_return = types.void
    if not sync:
        ty_event = DpctlSyclEvent()
        ty_return = types.Tuple([ty_event, ty_event])

    if ty_dependent_events is not None:
        if not isinstance(ty_dependent_events, UniTuple) and not isinstance(
            ty_dependent_events, Tuple
        ):
            raise ValueError("dependent events must be passed as a tuple")

        sig = ty_return(
            ty_kernel_fn,
            ty_index_space,
            ty_dependent_events,
            ty_kernel_args_tuple,
        )
    else:
        sig = ty_return(ty_kernel_fn, ty_index_space, ty_kernel_args_tuple)

    # Add Item/NdItem as a first argument to kernel arguments list. It is
    # an empty struct so any other modifications at kernel submission are not
    # needed.
    if len(signature(ty_kernel_fn.dispatcher.py_func).parameters) > len(
        ty_kernel_args_tuple
    ):
        if isinstance(ty_index_space, RangeType):
            ty_item = ItemType(ty_index_space.ndim)
        else:
            ty_item = NdItemType(ty_index_space.ndim)

        ty_kernel_args_tuple = (ty_item, *ty_kernel_args_tuple)
    else:
        warnings.warn(
            "Kernels without item/nd_item will be not supported in the future",
            DeprecationWarning,
        )

    # ty_kernel_fn is type specific to exact function, so we can get function
    # directly from type and compile it. Thats why we don't need to get it in
    # codegen
    kernel_dispatcher: SPIRVKernelDispatcher = ty_kernel_fn.dispatcher
    kcres: _SPIRVKernelCompileResult = kernel_dispatcher.get_compile_result(
        types.void(*ty_kernel_args_tuple)  # kernel signature
    )
    kernel_module: kl.SPIRVKernelModule = kcres.kernel_device_ir_module
    kernel_targetctx: SPIRVTargetContext = kernel_dispatcher.targetctx

    def codegen(
        cgctx: CPUContext, builder: llvmir.IRBuilder, sig, llargs: list
    ):
        ty_index_space: Union[RangeType, NdRangeType] = sig.args[1]
        ll_index_space: llvmir.Instruction = llargs[1]
        ty_kernel_args_tuple: UniTuple = sig.args[-1]
        ll_kernel_args_tuple: llvmir.Instruction = llargs[-1]

        if len(llargs) == 4:
            ty_dependent_events: UniTuple = sig.args[2]
            ll_dependent_events: llvmir.Instruction = llargs[2]
        else:
            ty_dependent_events = None

        kl_builder = kl.KernelLaunchIRBuilder(
            cgctx,
            builder,
            kernel_targetctx.data_model_manager,
        )
        kl_builder.set_range_from_indexer(
            ty_indexer_arg=ty_index_space,
            ll_index_arg=ll_index_space,
        )
        kl_builder.set_arguments_form_tuple(
            ty_kernel_args_tuple, ll_kernel_args_tuple
        )
        kl_builder.set_queue_from_arguments()
        kl_builder.set_kernel_from_spirv(kernel_module)
        if ty_dependent_events is None:
            kl_builder.set_dependent_events([])
        else:
            kl_builder.set_dependent_events_from_tuple(
                ty_dependent_events,
                ll_dependent_events,
            )
        device_event_ref = kl_builder.submit()

        if not sync:
            host_event_ref = kl_builder.acquire_meminfo_and_submit_release()

            return wrap_event_reference_tuple(
                cgctx,
                builder,
                wrap_event_reference(cgctx, builder, host_event_ref),
                wrap_event_reference(cgctx, builder, device_event_ref),
            )

        sycl.dpctl_event_wait(builder, device_event_ref)
        sycl.dpctl_event_delete(builder, device_event_ref)

        return None

    return sig, codegen


@dpjit
def call_kernel(kernel_fn, index_space, *kernel_args) -> None:
    """Calls a numba_dpex.kernel decorated function from CPython or from another
    dpjit function. Kernel execution happens in synchronous way, so the thread
    will be blocked till the kernel done execution.

    Args:
        kernel_fn (numba_dpex.experimental.KernelDispatcher): A
        numba_dpex.kernel decorated function that is compiled to a
        KernelDispatcher by numba_dpex.
        index_space (Range | NdRange): A numba_dpex.Range or numba_dpex.NdRange
        type object that specifies the index space for the kernel.
        kernel_args : List of objects that are passed to the numba_dpex.kernel
        decorated function.
    """
    _submit_kernel_sync(  # pylint: disable=E1120
        kernel_fn,
        index_space,
        kernel_args,
    )


@dpjit
def call_kernel_async(
    kernel_fn,
    index_space,
    dependent_events: list[dpctl.SyclEvent],
    *kernel_args
) -> tuple[dpctl.SyclEvent, dpctl.SyclEvent]:
    """Calls a numba_dpex.kernel decorated function from CPython or from another
    dpjit function. Kernel execution happens in asynchronous way, so the thread
    will not be blocked till the kernel done execution. That means that it is
    user responsibility to properly use any memory used by kernel until the
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
    return _submit_kernel_async(  # pylint: disable=E1120
        kernel_fn,
        index_space,
        dependent_events,
        kernel_args,
    )
