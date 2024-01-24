# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import cgutils

from numba_dpex.core.runtime import context as dpexrt
from numba_dpex.core.types import DpctlSyclEvent


def wrap_event_reference(ctx, builder, eref):
    """Wrap dpctl event reference into datamodel so it can be boxed to
    Python."""

    ty_event = DpctlSyclEvent()

    pyapi = ctx.get_python_api(builder)

    event_struct_proxy = cgutils.create_struct_proxy(ty_event)(ctx, builder)

    # Ref count after the call is equal to 1.
    # TODO: get dpex RT from cached property once the PR is merged
    # https://github.com/IntelPython/numba-dpex/pull/1027
    # ctx.dpexrt.eventstruct_init( # noqa: W0621
    dpexrt.DpexRTContext(ctx).eventstruct_init(
        pyapi,
        eref,
        # calling _<method>() is by numba's design
        event_struct_proxy._getpointer(),  # pylint: disable=W0212
    )

    # calling _<method>() is by numba's design
    event_value = event_struct_proxy._getvalue()  # pylint: disable=W0212

    return event_value
