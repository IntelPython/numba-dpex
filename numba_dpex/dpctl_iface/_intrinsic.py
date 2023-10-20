# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.datamodel import default_manager
from numba.extending import intrinsic, overload_method

import numba_dpex.dpctl_iface.libsyclinterface_bindings as sycl
from numba_dpex.core import types as dpex_types


@intrinsic
def sycl_event_wait(typingctx, ty_event: dpex_types.DpctlSyclEvent):
    sig = types.void(dpex_types.DpctlSyclEvent())

    # defines the custom code generation
    def codegen(context, builder, signature, args):
        sycl_event_dm = default_manager.lookup(ty_event)
        event_ref = builder.extract_value(
            args[0],
            sycl_event_dm.get_field_position("event_ref"),
        )

        sycl.dpctl_event_wait(builder, event_ref)

    return sig, codegen


@overload_method(dpex_types.DpctlSyclEvent, "wait")
def ol_dpctl_sycl_event_wait(
    event,
):
    """Implementation of an overload to support dpctl.SyclEvent() inside
    a dpjit function.
    """
    return lambda event: sycl_event_wait(event)


# We don't want user to call sycl_event_wait(event), instead it must be called
# with event.wait(). In that way we guarantee the argument type by the
# @overload_method.
__all__ = []
