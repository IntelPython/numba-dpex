# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
from llvmlite.ir import IRBuilder
from numba import types
from numba.extending import intrinsic, overload, overload_method

import numba_dpex.dpctl_iface.libsyclinterface_bindings as sycl
from numba_dpex.core import types as dpex_types
from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME
from numba_dpex.dpctl_iface.wrappers import wrap_event_reference


@intrinsic(target=DPEX_TARGET_NAME)
def sycl_event_create(
    ty_context,
):
    """A numba "intrinsic" function to inject dpctl.SyclEvent constructor code.

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """
    ty_event = dpex_types.DpctlSyclEvent()

    sig = ty_event(types.void)

    def codegen(context, builder: IRBuilder, sig, args: list):
        event = sycl.dpctl_event_create(builder)
        return wrap_event_reference(context, builder, event)

    return sig, codegen


@intrinsic(target=DPEX_TARGET_NAME)
def sycl_event_wait(typingctx, ty_event: dpex_types.DpctlSyclEvent):
    sig = types.void(dpex_types.DpctlSyclEvent())

    # defines the custom code generation
    def codegen(context, builder, signature, args):
        sycl_event_dm = context.data_model_manager.lookup(ty_event)
        event_ref = builder.extract_value(
            args[0],
            sycl_event_dm.get_field_position("event_ref"),
        )

        sycl.dpctl_event_wait(builder, event_ref)

    return sig, codegen


@overload(dpctl.SyclEvent, target=DPEX_TARGET_NAME)
def ol_dpctl_sycl_event_create():
    """Implementation of an overload to support dpctl.SyclEvent() inside
    a dpjit function.
    """
    return lambda: sycl_event_create()


@overload_method(dpex_types.DpctlSyclEvent, "wait", target=DPEX_TARGET_NAME)
def ol_dpctl_sycl_event_wait(
    event,
):
    """Implementation of an overload to support dpctl.SyclEvent.wait() inside
    a dpjit function.
    """
    return lambda event: sycl_event_wait(event)


# We don't want user to call sycl_event_wait(event), instead it must be called
# with event.wait(). In that way we guarantee the argument type by the
# @overload_method.
__all__ = []
