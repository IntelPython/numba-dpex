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

import dpnp
from numba import types
from numba.extending import lower_builtin


@lower_builtin(dpnp.zeros, types.Any, types.Any, types.Any, types.Any)
def impl_dpnp_zeros(context, builder, sig, args):
    """
    Inputs: shape, dtype, usm_type, queue
    """
    from numba.core.imputils import impl_ret_new_ref

    from .dpnp_empty import _empty_nd_impl, _parse_empty_args

    empty_args = _parse_empty_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, *empty_args)
    _zero_fill_nd(context, builder, ary)
    return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())


def _zero_fill_nd(context, builder, ary):
    args = ary.data, builder.mul(ary.itemsize, ary.nitems), 0, ary.queue
    _memset(context, builder, *args)


def _memset(context, builder, ptr, size, value, queue):
    """
    Fill *size* bytes starting from *ptr* with *value*.
    """
    from numba.core.cgutils import int8_t, voidptr_t

    from numba_dppy.dpctl_iface.dpctl_capi_fn_builder import (
        DpctlCAPIFnBuilder as dpctl_api,
    )

    memset_fn = dpctl_api.get_dpctl_queue_memset(builder, context)
    event_wait_fn = dpctl_api.get_dpctl_queue_wait(builder, context)

    ptr = builder.bitcast(ptr, voidptr_t)
    if isinstance(value, int):
        value = int8_t(value)

    event_ref = builder.call(memset_fn, queue, ptr, size, value)
    builder.call(event_wait_fn, [event_ref])


# TODO: implement _ones_fill_nd
